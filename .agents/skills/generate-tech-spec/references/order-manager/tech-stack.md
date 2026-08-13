# Technical Specification: order_manager

## Assumptions

- **NFR Tier**: M (Medium) — financial-domain heuristic applied
- **Throughput**: ~500 orders/sec peak, ~100 orders/sec sustained
- **Latency budget (p99)**: <200 ms for order intake (request → 200 OK with order_id); settlement workflow latency unbounded (driven by external settlement event)
- **Consistency**: Strong on the hot path (account state, balance check, pending_orders write); strong on the settlement transactional move (single Aurora transaction)
- **RPO / RTO**: RPO 1h / RTO 30 min (Tier M defaults). Aurora point-in-time recovery enabled.
- **Compliance**: SOX, FINRA-adjacent — audit trail mandatory, immutable transaction history, encryption at rest with customer-managed KMS
- **Region strategy**: Single region, multi-AZ (us-east-1, three AZs). Multi-region DR is out of scope for v1.
- **Other assumptions**:
  - The downstream broker / exchange gateway exposes a synchronous submission API and emits asynchronous settlement events (push or polled).
  - Order IDs are client-generated UUIDs to enable end-to-end idempotency.
  - Per-account order ordering must be preserved; cross-account ordering is not required.

> If any assumption is wrong for your context, fix it here and regenerate downstream artifacts.

## Workload Archetype

**Hybrid: Sync OLTP API (intake) + Async event-driven workflow (settlement).**

Subsystem mapping:
- **Order intake** is sync OLTP — the user gets a definitive accept/reject response with an order_id within the latency budget.
- **Order submission to broker** is sync from the workflow's perspective but is buffered through SQS FIFO so we can absorb spikes and preserve per-account ordering.
- **Settlement** is async event-driven — an external settlement event triggers a workflow that performs the atomic state move from pending to history.

## Language & Runtime

- **Language**: Python 3.12
- **Rationale**: I/O-bound validation fan-out (account, trade-type, position, balance lookups) benefits from `asyncio` concurrency. Strong AWS SDK (`boto3`, `aioboto3`) ecosystem. Team familiarity reduces operational cost. Cold-start latency on Lambda (~100–300 ms with provisioned concurrency or SnapStart-equivalent strategies) fits the latency budget.
- **Well-Architected pillars served**: Operational Excellence (team familiarity), Performance Efficiency (asyncio fan-out), Cost Optimization (smaller memory footprint than Java).

## Compute

- **Order intake API**: AWS Lambda (Python 3.12, 1024 MB, provisioned concurrency = 20 baseline, auto-scaled by ALIAS) behind API Gateway HTTP API.
  - Rationale: Spiky workload, pay-per-request economics dominate at this scale; provisioned concurrency on the alias eliminates cold-start tail latency for the p99 budget.
- **Validation orchestration**: Step Functions Express Workflow.
  - Rationale: Sub-5-minute, high-volume orchestration; Express pricing is appropriate; durable state transitions and retry semantics for free.
- **Validators (4 functions)**: AWS Lambda — `account_validator`, `trade_type_validator`, `position_fetcher`, `balance_validator`. Run as a Step Functions `Parallel` state.
  - Rationale: Independent IO-bound checks parallelize naturally.
- **Order submitter**: AWS Lambda triggered by SQS FIFO. Calls the broker / exchange submission API and writes to `pending_orders`.
  - Rationale: SQS FIFO with `MessageGroupId=account_id` preserves per-account ordering; Lambda concurrency is naturally bounded by `MessageGroupId` count.
- **Settlement workflow**: Step Functions Standard (durable, long-running, exactly-once-per-event).
  - Rationale: Settlement events may arrive minutes to days after submission; Standard's 1-year max execution covers any realistic delay.
- **Well-Architected pillars served**: Reliability (Step Functions retries + durable state), Performance Efficiency (parallel validation), Cost Optimization (serverless pay-per-use).

## Concurrency & Communication

- **Sync paths**:
  - Client → API Gateway HTTP API (HTTPS, mTLS optional) → `order_intake` Lambda
  - `order_intake` Lambda → Step Functions Express (StartSyncExecution) for validation
  - Validators → ElastiCache (Valkey RESP) for account state and position cache
  - Validators → Aurora PostgreSQL (read replica) for authoritative balance / position lookups when cache miss
- **Async paths**:
  - Validation success → SQS FIFO `order_submission_queue.fifo` (MessageGroupId = account_id, MessageDeduplicationId = order_id) — at-least-once delivery, FIFO ordering per account
  - Settlement event source → EventBridge custom bus `order-events` → Step Functions Standard (`settlement_workflow`)
  - State transitions → EventBridge events: `order.placed`, `order.submitted`, `order.settled`, `order.rejected`
- **Idempotency strategy**:
  - Client-generated `order_id` (UUID) is the idempotency key end-to-end.
  - SQS deduplication via `MessageDeduplicationId = order_id` (5-minute window).
  - Aurora upserts use `INSERT ... ON CONFLICT (order_id) DO NOTHING` for `pending_orders`.
  - Settlement workflow keys on `order_id` to make the atomic move idempotent: the move from `pending_orders` to `transaction_history` is wrapped in a single `BEGIN; INSERT INTO transaction_history ... SELECT FROM pending_orders WHERE order_id = $1; DELETE FROM pending_orders WHERE order_id = $1; COMMIT;` — replays of the settlement event are no-ops once the row has moved.

## Storage

- **Operational DB**: Aurora PostgreSQL 15 (r6g.large, multi-AZ, one writer + one reader).
  - Reason: Multi-row transactional consistency required for the settlement move (`pending_orders` → `transaction_history`). Reporting and reconciliation queries (joins across accounts, positions, transactions) are first-class in Postgres. Point-in-time recovery satisfies the 1h RPO.
- **Cache**: ElastiCache (Valkey 7.x, cache.r6g.large, multi-AZ replication group).
  - Reason: Sub-millisecond reads on hot account-state and position keys reduce p99 of the validation fan-out by ~50–80 ms. Valkey is the default OSS Redis-compatible engine going forward.
- **Object storage**: S3 with **Object Lock in Compliance mode**, lifecycle to S3 Glacier Instant Retrieval at 90 days.
  - Reason: Audit trail of every order request, validation outcome, broker response, and settlement event. Object Lock satisfies regulatory immutability requirements (SOX, FINRA WORM-equivalent).
- **Search / vector / time-series**: Not applicable for this workload.
- **Access patterns covered**:
  - "Get account state by account_id" (cache-first, Aurora fallback)
  - "Get current positions for account_id" (cache-first)
  - "Insert pending order" (Aurora, single-row, idempotent)
  - "Atomically move pending order to history" (Aurora transaction)
  - "List pending orders for account_id" (Aurora index on `(account_id, status)`)
  - "Reconcile transactions for date range" (Aurora reader replica)
  - "Retrieve immutable audit record by order_id" (S3)

## Messaging & Events

- **Queues**: SQS FIFO `order_submission_queue.fifo`. Ordering: per-`account_id` via `MessageGroupId`. Delivery: at-least-once with deduplication on `order_id` (5-min window). Dead-letter queue `order_submission_dlq.fifo` after 5 receive attempts.
- **Event bus**: EventBridge custom bus `order-events`. Domain events: `order.placed`, `order.submitted`, `order.settled`, `order.rejected`, `order.cancelled`. Schema registry enforced via EventBridge Schemas.
- **Streaming**: Not applicable for this workload.

## Orchestration

- **Workflow engine**: Step Functions — Express for validation hot path, Standard for settlement.
- **Critical workflows**:
  - `order_validation_workflow` (Express): Parallel( account_validator, trade_type_validator, position_fetcher → balance_validator ) → Aggregate → Pass/Fail decision → enqueue to SQS or return rejection. Target p99 <100 ms.
  - `settlement_workflow` (Standard): Triggered by `order.settled` event → Aurora transaction (move row) → S3 audit write → emit `order.settled` confirmation. Idempotent on `order_id`.

## Data Engineering

Not applicable for this workload. Reporting and reconciliation queries run directly against the Aurora reader replica. If reporting volume grows, add a downstream pipeline (DMS → S3 → Glue → Athena) in v2.

## API Layer

- **Edge**: CloudFront in front of API Gateway for TLS termination at PoP and basic geographic blocking. Skip if all traffic is internal/partner via PrivateLink.
- **Gateway**: API Gateway HTTP API (cheaper, lower latency than REST API; we don't need request validation at the edge — Lambda does it).
- **Auth**: JWT authorizer on API Gateway, validating tokens issued by the existing IdP (Cognito or external OIDC). Per-account scope claims required for order-placement endpoints.
- **Rate limiting**: API Gateway usage plans + per-account token-bucket in Lambda backed by ElastiCache. Default 50 orders/sec/account, configurable.

## Security & IAM

- **Encryption at rest**: Customer-managed KMS CMK (`alias/order-manager`) for Aurora cluster, ElastiCache replication group, S3 audit bucket, SQS, and Lambda environment variables. CloudHSM not required for v1.
- **Encryption in transit**: TLS 1.2+ everywhere. mTLS optional between API Gateway and partner integrations.
- **Secrets**: AWS Secrets Manager — Aurora master credentials, broker API credentials, IdP signing keys (cached). 30-day automatic rotation for DB credentials.
- **Network**: Single VPC, three AZs. Two private subnet tiers — application (Lambda, Step Functions targets) and data (Aurora, ElastiCache). VPC endpoints (Interface) for SQS, EventBridge, Secrets Manager, KMS, CloudWatch Logs, S3 (Gateway endpoint). No NAT Gateway required for AWS-service egress.
- **IAM**: One execution role per Lambda, scoped to the minimum API actions on the minimum resource ARNs. Step Functions role scoped to the specific Lambda ARNs and EventBridge bus ARN. No `*` resources on data-plane policies.
- **Audit**: CloudTrail (org-wide, multi-region), AWS Config rules for "no public S3", "encryption enforced", "no IAM `*:*`". S3 Object Lock in Compliance mode on the audit bucket. VPC Flow Logs to S3.

## Observability

- **Metrics** (key SLIs):
  - `order_intake.p99_latency_ms` — alarm at 200 ms
  - `order_intake.error_rate` — alarm at 1%
  - `validation.failure_rate_by_reason` — informational, dimensioned by reason
  - `settlement.lag_seconds` — time from `order.submitted` to `order.settled`; alarm at p95 > 1h
  - `sqs.fifo.age_oldest_message` — alarm at 60 s
  - `aurora.cpu_utilization`, `aurora.replica_lag`, `aurora.connections_in_use`
- **Logs**: One CloudWatch log group per Lambda, 90-day retention, JSON-structured (`order_id`, `account_id`, `trace_id`, `event`, `outcome`, `duration_ms`). Single schema across services.
- **Traces**: X-Ray enabled across API Gateway → Lambda → Step Functions → Lambda → Aurora. Sampling 10% on success, 100% on error.
- **Alarms**:
  - `order_intake.p99_latency_ms > 200ms for 3 consecutive minutes` → SNS → PagerDuty
  - `order_intake.error_rate > 1% for 5 minutes` → SNS → PagerDuty
  - `sqs.dlq.message_count > 0` → SNS → PagerDuty
  - `settlement.lag_seconds.p95 > 3600s` → SNS → email (warn, not page)
  - `aurora.replica_lag > 5s` → SNS → email
- **Dashboards**: One CloudWatch dashboard `order-manager-prod` with intake latency, error rate, SQS depth, Aurora load, settlement lag.

## Estimated Cost Profile

Order-of-magnitude monthly cost at the assumed scale (~500 orders/sec peak, ~100 sustained, ~260M orders/month):

| Service | Estimate | Notes |
|---|---|---|
| Aurora PostgreSQL r6g.large multi-AZ | ~$400 | 1 writer + 1 reader, reserved instance discount available |
| Lambda (intake + validators + submitter + settlement) | ~$300 | At ~100 orders/sec sustained, ~6× Lambda invocations per order |
| Step Functions Express | ~$400 | ~260M state-machine executions / month |
| Step Functions Standard (settlement) | ~$50 | ~260M state transitions, lower volume than Express |
| ElastiCache cache.r6g.large multi-AZ | ~$200 | 2 nodes |
| API Gateway HTTP API | ~$100 | At 260M requests/month |
| SQS FIFO | ~$50 | 260M requests, FIFO premium |
| EventBridge custom bus | ~$30 | Few hundred million events |
| S3 + Object Lock + Glacier IR | ~$100 | Audit trail growth ~50 GB/month, retention 7y |
| CloudWatch (logs + metrics + X-Ray) | ~$200 | Log ingestion dominates |
| KMS, Secrets Manager, misc | ~$70 | |
| **Total** | **~$1,900–2,300** | |

**Top 3 cost drivers**: Step Functions Express, Aurora, Lambda. Cost optimizations to consider in v2: Lambda SnapStart on Java if a rewrite is justified (unlikely), Aurora reserved instances (~30% savings), CloudWatch Logs retention shortening + selective sampling.

## Rejected Alternatives

- **Compute alternative**: ECS Fargate for the intake API — Rejected because workload is spiky and per-request economics favor Lambda; Fargate adds container orchestration ops overhead with no clear win at this scale.
- **Database alternative**: DynamoDB for `pending_orders` and `transaction_history` — Rejected because multi-row transactional consistency for the settlement move is harder (TransactWriteItems works but is more brittle than a single Aurora transaction), and reporting/reconciliation queries with joins are first-class in Postgres. DynamoDB would be the right call if access patterns were strictly key-lookup and write volume was 10x higher.
- **Messaging alternative**: Kinesis Data Streams for order submission — Rejected because we don't need replay or fan-out to multiple consumers, and per-account ordering via MessageGroupId on SQS FIFO is simpler than partition-key engineering on Kinesis. Kinesis would win if we added a real-time analytics consumer.
- **Orchestration alternative**: Custom Lambda chain (no Step Functions) — Rejected because we'd reimplement retries, error handling, and observability that Step Functions provides for free, and we'd lose visual workflow tracing.
- **Language alternative**: Java 21 — Rejected because Lambda cold-start cost and per-invocation memory footprint dominate at this scale; Java's throughput advantage matters in long-running services, not per-request Lambda. SnapStart narrows the gap but the team-familiarity tilt favors Python.

## Well-Architected Summary

- **Operational Excellence**: Step Functions provides visual workflow tracing; structured JSON logs across services share one schema; X-Ray spans the full request path; CloudWatch dashboard concentrates the SLIs in one place.
- **Security**: Customer-managed KMS CMKs for all data-at-rest; Secrets Manager (no env-var credentials); private-only VPC topology with VPC endpoints; IAM least-privilege per function; S3 Object Lock for regulatory audit immutability; WAF on the API edge.
- **Reliability**: Multi-AZ Aurora and ElastiCache; SQS FIFO durability for in-flight orders; Step Functions retries with exponential backoff; idempotent design via client-generated `order_id`; DLQ on the order-submission queue; settlement workflow is idempotent on `order_id` so replayed events are no-ops.
- **Performance Efficiency**: Validators run in parallel via Step Functions Parallel state; ElastiCache absorbs hot reads; provisioned concurrency on the intake Lambda eliminates cold-start tail latency; HTTP API (not REST API) for lower latency.
- **Cost Optimization**: Serverless pay-per-use for compute; HTTP API instead of REST API; S3 lifecycle to Glacier Instant Retrieval after 90 days; reserved-instance candidate for Aurora; CloudWatch log retention bounded at 90 days for non-audit streams.
- **Sustainability**: Serverless tiers idle to zero; Graviton (r6g, arm64 Lambda) where supported; cache layer reduces redundant database compute.
