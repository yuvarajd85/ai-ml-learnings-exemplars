# Technical Specification: Insurance Claim Intake & Adjuster Workspace

## Assumptions
- **NFR Tier**: M (Medium) — mid-size insurer; regulatory domain elevates baseline from S
- **Throughput**: 200–800 API req/s peak; 300–1,000 claims/day; 1,500–5,000 document uploads/day
- **Latency budget (p99)**: <200 ms for adjuster search and claim detail; <500 ms for submission acknowledgment; extraction pipeline SLA <15 min end-to-end
- **Consistency**: Strong on claim status transitions (ACID); eventual for search index and dashboard aggregates
- **RPO / RTO**: RPO 1 h / RTO 30 min
- **Compliance**: State insurance-commissioner regulation — 7-year WORM retention, per-claim audit trail, subpoena response ≤30 days; treat as financial-adjacent (audit trail with S3 Object Lock, KMS CMK, no public buckets)
- **Region strategy**: Single-region (us-east-1) multi-AZ; no active-passive DR at this scale — RPO/RTO met by Aurora automated backups + S3 versioning
- **Other assumptions**:
  - Customer and adjuster authentication required; adjusters are internal employees (IdP via Cognito + corporate SSO federation)
  - Mobile push notifications delivered via iOS APNs and Android FCM through SNS
  - Payment disbursement handled by downstream system via API call — not designed here
  - Average document: 5 pages, 3 MB; 1–10 documents per claim; mix of PDFs, JPEGs, PNGs
  - Management dashboards refreshed daily; finance reports generated monthly
  - Team has moderate AWS familiarity; no Kubernetes operational maturity assumed

> If any assumption is wrong for your context, fix it here and regenerate downstream artifacts.

## Workload Archetype
**Hybrid**: Sync OLTP API + Async event-driven + Batch ETL

- **Claim submission** (web/mobile form + document upload): Sync OLTP API — customer waits for submission acknowledgment, then documents upload asynchronously to S3
- **Document ingestion & AI extraction**: Async event-driven — S3 upload event triggers Step Functions workflow (Textract → Bedrock classification → Aurora write → OpenSearch index)
- **Triage & routing**: Async event-driven — Step Functions workflow triggered on extraction completion; runs adjuster assignment logic
- **Adjuster workspace** (search, status update, document view): Sync OLTP API
- **Customer notifications**: Async event-driven — EventBridge domain events fan out to SNS (email + push)
- **Management dashboards & finance reports**: Batch ETL — EventBridge Scheduler triggers nightly Glue job; QuickSight reads from Redshift Serverless

## Language & Runtime
- **Language**: Python 3.12
- **Rationale**: Mature AWS SDK (boto3) support for Textract, Bedrock, Step Functions, Aurora Data API, and OpenSearch; strong LangChain/AI ecosystem for future enhancements; team is already Python-proficient per codebase
- **Well-Architected pillars served**: Operational Excellence (single language across Lambda functions reduces cognitive overhead), Performance Efficiency (Python 3.12 SnapStart equivalent via Lambda function URLs warm containers)

## Compute
- **Claim Submission API (Lambda)**: AWS Lambda (Python 3.12, 512 MB, 30s timeout) behind API Gateway HTTP API
  - Rationale: Claim submissions are spiky (business hours peak), stateless, and sub-15s. Lambda eliminates idle cost for off-hours. Each submission creates the claim record in Aurora and returns pre-signed S3 URLs — well within Lambda's execution model.
- **Adjuster Workspace API (Lambda)**: AWS Lambda (Python 3.12, 1024 MB, 15s timeout) behind API Gateway HTTP API
  - Rationale: Adjuster API handles search (OpenSearch query), claim detail fetch (Aurora), status update, and S3 pre-signed URL generation. All request-scoped, <15 min, benefits from Lambda's concurrency model without persistent connections.
- **Document Ingestion Workflow**: AWS Step Functions Standard workflow driving Lambda functions (512 MB, 300s timeout each: file-type-detector, textract-poller, bedrock-classifier, db-writer, search-indexer)
  - Rationale: Standard (not Express) because extraction pipelines exceed 5 minutes for large multi-page documents; Standard provides durable execution history required for audit; built-in retry and error-path visibility via CloudWatch.
- **Triage & Routing Workflow**: AWS Step Functions Standard workflow driving Lambda functions (512 MB, 60s timeout: complexity-classifier, adjuster-assigner, notification-emitter)
  - Rationale: Chained from ingestion workflow via EventBridge; durable, retryable, observable via Step Functions console — critical for operations SLA tracking.
- **Nightly ETL (Glue)**: AWS Glue 4.0 Spark job (G.1X workers, auto-scaling 2–10 workers)
  - Rationale: Daily aggregation of claim events from Aurora to Redshift Serverless for BI. Glue handles Aurora JDBC source, S3 staging, and Redshift COPY natively; runs once per day, no persistent compute needed.
- **Well-Architected pillars served**: Cost Optimization (Lambda + Glue eliminate idle compute), Reliability (Step Functions retry/error paths), Operational Excellence (Step Functions execution visibility)

## Concurrency & Communication
- **Sync paths**:
  - Customer → CloudFront → WAF → API Gateway HTTP API → Lambda (claim intake): HTTPS/REST
  - Adjuster → API Gateway HTTP API → Lambda (adjuster API) → Aurora / OpenSearch: HTTPS/REST
  - Lambda (adjuster API) → S3 (pre-signed GET URL for document view): HTTPS
  - Triage workflow → Payment platform API: HTTPS/REST (status-only webhook, no disbursement)
- **Async paths**:
  - S3 (document upload complete) → SQS Standard queue (document-intake-queue) → EventBridge Pipes → Step Functions ingestion workflow trigger: at-least-once delivery; Lambda deduplicates on S3 ETag + claim_id composite key
  - Step Functions ingestion workflow → EventBridge custom bus (insurance-domain-events) → "extraction.completed" event: at-least-once
  - EventBridge → Step Functions triage workflow (via EventBridge Pipes): at-least-once
  - EventBridge → SNS topic (claim-status-notifications) → SES (email) + SNS Mobile Push (iOS APNs / Android FCM): at-least-once; push is best-effort per FCM/APNs contract
  - Glue nightly job → Redshift Serverless (COPY from S3): batch, exactly-once per run
- **Idempotency strategy**: Each Lambda writes a `processed_at` timestamp and `idempotency_key` (S3 object key for ingestion; `{claim_id}:{status}:{timestamp}` for status transitions) checked against a DynamoDB idempotency table (TTL 24h) before any Aurora write or EventBridge publish

## Storage
- **Operational DB**: Aurora PostgreSQL 16 (r6g.large, Multi-AZ, 2 read replicas for adjuster search offload)
  - Reason: ACID state machine for claim status transitions; complex joins for adjuster workload queries; native full-text search for narrative fields (supplemented by OpenSearch for document-level search); Point-In-Time Recovery covers RPO 1h; Aurora Global Database available if active-passive DR added later
- **Cache**: Amazon ElastiCache for Valkey (cache.t4g.medium, single-AZ — adjuster workload metadata, policy lookup cache, claims-in-flight TTL 5 min)
  - Reason: Adjuster workspace frequently re-fetches policy details and adjuster pool assignments; caching eliminates repeated Aurora reads; Valkey preferred over Redis OSS for new builds
- **Object storage**: Amazon S3 (us-east-1)
  - `claims-documents-{account-id}` bucket: S3 Object Lock in Compliance mode, 7-year retention period, KMS CMK encryption — stores all uploaded documents, Textract output JSON
  - `claims-audit-log-{account-id}` bucket: S3 Object Lock in Compliance mode, 7-year retention — receives immutable per-claim event streams (status transitions, adjuster decisions, extraction results) as newline-delimited JSON objects; supports subpoena export via Lambda aggregator
  - `claims-analytics-{account-id}` bucket: daily Glue ETL output (Parquet, partitioned by date/claim-type); Athena-queryable for ad-hoc finance queries
  - S3 Intelligent-Tiering on documents bucket (transition to Glacier after 180 days, Deep Archive after 730 days)
- **Search / vector / time-series**: Amazon OpenSearch Serverless collection (OCU auto-scaling, `insurance-claims` index)
  - Reason: Full-text search over claim narratives and extracted document text; structured filters on policy number, claimant name, date range, claim type, adjuster ID; serverless eliminates capacity planning for variable adjuster search load
- **Access patterns covered**:
  - Claim status read/write by claim_id (Aurora PK lookup)
  - Adjuster workload query: open claims by adjuster_id, ordered by received_at (Aurora indexed query, offloaded to read replica)
  - Full-text search across narrative + extracted text by keyword, date range, policy number, claimant name (OpenSearch)
  - Document retrieval by claim_id + document_id → S3 pre-signed URL (Aurora foreign key → S3 key lookup)
  - Audit log retrieval per claim for subpoena (S3 prefix query by claim_id)
  - Dashboard aggregates: claim volume by type/date, avg time-in-stage, payout trends (Redshift Serverless)

## Messaging & Events
- **Queues**:
  - `document-intake-queue` (SQS Standard): receives S3 event notifications for new document uploads; visibility timeout 600s; DLQ `document-intake-dlq` after 3 retries; at-least-once delivery
  - `claim-status-dlq` (SQS Standard): dead-letter for failed EventBridge → Step Functions triage triggers
- **Event bus**: Amazon EventBridge custom bus `insurance-domain-events`
  - Events: `claim.submitted`, `extraction.completed`, `claim.triaged`, `claim.status.changed`, `claim.assigned`
  - Rules: `extraction.completed` → triggers triage Step Functions via EventBridge Pipes; `claim.status.changed` → SNS topic `claim-status-notifications`
- **Streaming**: Not applicable for this workload — claim volume does not require Kinesis; SQS + EventBridge sufficient

## Orchestration
- **Workflow engine**: AWS Step Functions Standard (durable, long-running, audit history retained 90 days)
- **Critical workflows**:
  - `DocumentIngestionWorkflow`: S3 trigger → detect file type → start Textract async job → poll Textract completion (Wait state + callback token) → invoke Bedrock classifier → write Aurora + OpenSearch → emit `extraction.completed` to EventBridge; max duration 30 min
  - `TriageAndRoutingWorkflow`: triggered by `extraction.completed` → classify complexity (Bedrock) → query adjuster pool (Aurora read replica) → assign adjuster → update claim status → emit `claim.assigned`; max duration 5 min

## Data Engineering
- **Pipeline**: Daily ETL — EventBridge Scheduler (cron `0 2 * * *`) → triggers AWS Glue 4.0 Spark job
- **Stages**:
  1. Glue reads Aurora PostgreSQL via JDBC (claims, claim_events, adjuster_assignments tables)
  2. Transforms to aggregate dimensions: volume by claim_type/date, avg_duration_per_stage, adjuster_workload_distribution, payout_trend
  3. Writes Parquet to `claims-analytics-{account-id}` S3 bucket (partitioned by `dt=YYYY-MM-DD/claim_type=*`)
  4. Glue triggers Redshift Serverless COPY from S3 (incremental, append-only fact tables)
- **QuickSight**: SPICE refresh daily from Redshift Serverless; dashboards: claim-volume, time-in-stage, adjuster-workload, payout-trends
- **Finance monthly reports**: EventBridge Scheduler (monthly) → Lambda → Athena query over `claims-analytics` S3 → results emailed via SES as CSV attachment
- **Ad-hoc queries**: Athena + AWS Glue Data Catalog over `claims-analytics` S3

## API Layer
- **Edge**: Amazon CloudFront distribution in front of customer-facing API (claim submission + status check); WAF WebACL with AWS Managed Rules (AWSManagedRulesCommonRuleSet, AWSManagedRulesKnownBadInputsRuleSet)
- **Gateway**: Amazon API Gateway HTTP API (two APIs: `claims-customer-api` for submission/status; `claims-adjuster-api` for workspace — internal, no CloudFront)
- **Auth**:
  - Customer portal: Amazon Cognito User Pool with email/password + optional MFA; JWT authorizer on API Gateway
  - Adjuster portal: Amazon Cognito User Pool federated to corporate IdP via SAML 2.0 (IAM Identity Center SAML app); JWT authorizer on API Gateway; Cognito group-based authorization (adjuster / senior-adjuster / investigator / supervisor)
- **Rate limiting**: API Gateway HTTP API throttling — 100 req/s burst / 50 req/s steady per customer Cognito identity; 500 req/s aggregate per API; WAF rate-based rule 1,000 req/5-min per IP for customer API

## Security & IAM
- **Encryption at rest**: KMS CMK per data classification — `insurance/documents` CMK for S3 documents + Textract output; `insurance/db` CMK for Aurora + ElastiCache; `insurance/analytics` CMK for S3 analytics + Redshift; CMKs in us-east-1, key rotation enabled annually
- **Encryption in transit**: TLS 1.2 minimum everywhere; API Gateway enforces HTTPS; Aurora uses SSL-required parameter group; OpenSearch enforces node-to-node encryption + HTTPS-only; internal Lambda-to-Aurora via VPC (no public endpoint)
- **Secrets**: Secrets Manager for Aurora master credentials (auto-rotation 30 days), Textract/Bedrock do not require secrets (IAM role), OpenSearch master user credentials, corporate IdP SAML metadata URL
- **Network**:
  - VPC `10.0.0.0/16` with 3 AZs
  - Public subnets (`10.0.0.0/20` per AZ): NAT Gateways only; no compute
  - Private subnets (`10.0.16.0/20` per AZ): Lambda, Step Functions ENIs, Glue connections, ElastiCache
  - Isolated subnets (`10.0.32.0/20` per AZ): Aurora, OpenSearch
  - VPC endpoints: S3 (Gateway), DynamoDB (Gateway), Secrets Manager (Interface), Textract (Interface), Bedrock Runtime (Interface), SQS (Interface), EventBridge (Interface), Step Functions (Interface), CloudWatch Logs (Interface)
  - No public endpoints on Aurora, OpenSearch, or ElastiCache
- **IAM**: Separate execution role per Lambda function; no wildcard resource policies; Step Functions role scoped to specific Lambda ARNs and EventBridge bus ARN; Glue role scoped to specific S3 buckets and Aurora secret ARN; Cognito identity-pool roles scoped to presigned URL generation only (no direct S3 PutObject via SDK — Lambda handles pre-signing)
- **Audit**: AWS CloudTrail organization trail (all management + data events for S3 documents bucket and Aurora); CloudTrail logs to `claims-audit-log` S3 bucket with Object Lock; S3 Object Lock on both audit and documents buckets (Compliance mode, 7-year retention); per-claim event stream written to `claims-audit-log` S3 by every status-change Lambda (immutable, append-only, enables 30-day subpoena delivery via Lambda aggregator)

## Observability
- **Metrics** (key SLIs):
  - `ClaimSubmissionLatencyP99` — API Gateway + Lambda duration, target <500 ms
  - `ExtractionPipelineDuration` — Step Functions execution time, target <15 min
  - `AdjusterSearchLatencyP99` — Lambda + OpenSearch duration, target <200 ms
  - `TriageQueueDepth` — SQS `document-intake-queue` ApproximateNumberOfMessagesVisible, target <100
  - `ExtractionFailureRate` — Step Functions execution failures / total, alert >1%
  - `AdjusterWorkspaceErrorRate` — API Gateway 5XX / total, alert >0.5%
- **Logs**: CloudWatch Log Groups per Lambda function and Glue job; structured JSON (`{"timestamp","claim_id","event_type","duration_ms","status_code","user_id"}`); retention 90 days (Lambda/Step Functions), 1 year (audit-adjacent claim events); Lambda Powertools for structured logging + correlation IDs
- **Traces**: AWS X-Ray active tracing on all Lambda functions, API Gateway, and Step Functions; X-Ray service map spans: API Gateway → Lambda → Aurora / OpenSearch / Bedrock; sampling rate 5% baseline + 100% on errors
- **Alarms** (CloudWatch):
  - `ExtractionPipelineDuration > 20 min` → SNS ops-alert (P2)
  - `TriageQueueDepth > 200 messages` → SNS ops-alert (P1)
  - `AdjusterWorkspaceErrorRate > 0.5%` → SNS ops-alert (P2)
  - `ClaimSubmissionErrors > 10 in 5 min` → SNS ops-alert (P1)
  - `AuroraReplicaLag > 5 s` → SNS ops-alert (P2)
  - `DLQ document-intake-dlq visible messages > 0` → SNS ops-alert (P1)
- **Dashboards**: CloudWatch Dashboard `InsuranceClaims-Operations` (real-time: API latencies, queue depths, Step Functions execution states, Aurora connections); CloudWatch Dashboard `InsuranceClaims-Business` (daily: claim volume, extraction success rate, triage latency distribution)

## Estimated Cost Profile
**Scale assumption**: 500 claims/day, 2,500 document uploads/day, avg 5 pages/doc, 50 adjuster users, 5 ops managers

| Service | Monthly estimate |
|---|---|
| AWS Textract (AnalyzeDocument, 375K pages/mo) | $3,750–$5,625 |
| Amazon Bedrock (Claude Haiku, ~16K invocations/mo) | $200–$400 |
| Aurora PostgreSQL r6g.large Multi-AZ + 2 read replicas | $450–$600 |
| OpenSearch Serverless (2 OCU indexing + 2 OCU search) | $350–$500 |
| API Gateway HTTP API + Lambda | $80–$150 |
| S3 (documents + audit + analytics, ~75 GB/mo new) | $50–$100 |
| Step Functions Standard (30K executions/mo) | $30–$60 |
| ElastiCache Valkey t4g.medium | $60–$80 |
| Amazon QuickSight (5 authors, SPICE) | $90–$150 |
| AWS Glue (nightly, G.1X × 5 DPU-hours/run) | $40–$80 |
| Redshift Serverless | $80–$150 |
| CloudWatch, X-Ray, CloudTrail, Secrets Manager, KMS | $100–$200 |
| **Total** | **~$5,300–$8,100 / month** |

**Top 3 cost drivers**:
1. **Textract** — dominates at 60–70% of total; AnalyzeDocument (forms+tables mode) priced per page. Mitigation: use DetectDocumentText (cheaper) for photo-only files; reserve AnalyzeDocument for PDFs with structured fields.
2. **Aurora PostgreSQL** — multi-AZ + read replicas are fixed cost. Mitigation: scale down read replicas during off-hours with Aurora Serverless v2 if usage is highly diurnal.
3. **OpenSearch Serverless** — minimum 2 OCU billing floor even at low load. Mitigation: acceptable at Tier M; revisit if claims volume drops below 100/day.

## Rejected Alternatives
- **Compute alternative — ECS Fargate for API layer**: Rejected because Lambda handles spiky claims-submission load at lower cost and operational complexity. ECS Fargate earns its place when persistent connections (WebSocket) or >15-min execution is required — neither applies to the submission or adjuster APIs.
- **Compute alternative — AWS Batch for document ingestion**: Rejected in favor of Step Functions + Lambda. Batch suits compute-heavy GPU/CPU workloads; Textract is a managed API call, not a CPU-intensive job. Step Functions provides superior execution visibility and built-in retry semantics for this orchestration pattern.
- **Database alternative — DynamoDB for claim records**: Rejected because adjuster search requires multi-predicate queries (policy number AND date range AND claimant name AND status), complex joins between claims and adjuster assignments, and reporting aggregations. DynamoDB's single-table access pattern is ill-suited to these read shapes without expensive GSI proliferation and application-side joins.
- **Search alternative — Aurora PostgreSQL full-text search**: Rejected for document-level search. Aurora pg_tsvector covers narrative text search adequately, but does not scale to cross-claim keyword search over extracted document text (thousands of pages). OpenSearch Serverless provides relevance ranking, highlighted snippets, and multi-field filter aggregations that adjusters expect.
- **Extraction alternative — Amazon Comprehend (entity extraction)**: Comprehend extracts named entities but does not parse structured form fields (policy number, VIN, itemized amounts) from PDFs. Textract's AnalyzeDocument handles form key-value pairs and tables natively; Bedrock handles the claim-type and complexity classification where prompt-based reasoning outperforms rule-based extraction.
- **Messaging alternative — Kinesis Data Streams**: Rejected in favor of SQS + EventBridge. Claim volume (2,500 uploads/day = ~0.03/s) does not justify Kinesis shard costs or operational overhead. SQS provides sufficient throughput with simpler consumer semantics and built-in DLQ.
- **Analytics alternative — Amazon Redshift provisioned cluster**: Rejected in favor of Redshift Serverless. Daily batch analytics at this scale does not justify a permanently running cluster. Serverless billing per RPU-second eliminates idle cost for nightly-only workloads.

## Well-Architected Summary
- **Operational Excellence**: Step Functions Standard provides full execution history and visual workflow tracing, enabling ops teams to diagnose extraction failures without bespoke logging. Lambda Powertools enforces structured JSON logging with correlation IDs. Two CloudWatch dashboards separate operational from business metrics. EventBridge decouples subsystems, allowing independent deployment and rollback.
- **Security**: KMS CMK per data classification (documents, DB, analytics) limits blast radius of key compromise. S3 Object Lock (Compliance mode) prevents accidental or malicious deletion of audit records — critical for regulatory defense. VPC endpoints for all internal AWS service calls eliminate NAT Gateway exposure and egress cost. IAM roles are scoped per Lambda function; no shared execution roles. Cognito SAML federation avoids storing adjuster credentials in AWS.
- **Reliability**: Aurora Multi-AZ with 2 read replicas provides automatic failover (<60s) and read scaling. SQS DLQ + Step Functions catch and preserve failed extraction events for replay — no claims are silently lost. DynamoDB idempotency table prevents duplicate claim records from SQS redelivery. S3 Object Lock + versioning protects documents from accidental overwrites. CloudWatch alarms with P1/P2 tiers enforce active incident response.
- **Performance Efficiency**: OpenSearch Serverless auto-scales indexing and search OCUs to match adjuster query patterns without over-provisioning. Lambda concurrency scales horizontally with claim submission spikes. ElastiCache Valkey eliminates repeated Aurora reads for hot adjuster-pool and policy metadata. Textract and Bedrock are managed APIs — no GPU/instance capacity to manage.
- **Cost Optimization**: Serverless compute (Lambda, Step Functions, Glue, OpenSearch Serverless, Redshift Serverless) eliminates idle cost for off-hours. S3 Intelligent-Tiering transitions documents to Glacier after 180 days and Deep Archive after 730 days — 7-year retention cost drops to ~$0.00099/GB-month for archived files. Textract cost mitigation (DetectDocumentText for photos) is the highest-leverage optimization. CloudFront caches static assets for the web portal, reducing API Gateway calls.
- **Sustainability**: Serverless-first architecture means compute scales to zero off-hours. Step Functions workflow duration is bounded by Textract response time, not polling loops — avoids waste from busy-wait. Glue auto-scaling (2–10 workers) matches actual ETL data volume. S3 archival reduces storage energy footprint over the 7-year retention window.
