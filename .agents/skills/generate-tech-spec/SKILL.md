---
name: generate-tech-spec
description: Generate a structured AWS technical specification and architecture diagram from a natural-language functional requirement. Use this skill whenever the user asks to "design a system for...", "architect a solution for...", "what stack should I use for...", "give me a tech spec...", "propose an architecture...", or describes a workload (trading, ETL, document processing, ingestion pipeline, API service, ML inference, etc.) and wants a recommended AWS technology stack. Trigger even when the user does not say "tech spec" — phrases like "how would you build this on AWS", "design the backend for...", or "what AWS services should I use for..." should also fire this skill. The output is two artifacts (`tech-stack.md` + an architecture diagram via the `ar-diagram` skill) intended to be consumed downstream by the `build-resources` skill, so the section structure of the tech-stack output is a contract — do not vary it.
allowed-tools: Read, Write, Edit, Glob, Bash
---

# Generate Tech Spec

## Purpose

Turn a functional requirement into a justified, AWS-centric technical specification anchored to the **AWS Well-Architected Framework**, plus a matching architecture diagram. The output is structured for downstream consumption by the `build-resources` skill, which generates IaC from these artifacts.

This skill is opinionated on purpose. Vague tech specs ("use Lambda or ECS") are useless to a build pipeline. Every recommendation must:
1. Name a specific service (e.g., "Aurora PostgreSQL r6g.large multi-AZ", not "Postgres on AWS").
2. Justify the choice against workload characteristics and Well-Architected pillars.
3. List at least one credible rejected alternative with reasoning.

## When to use this skill

**Use it when:**
- The user describes a system to be built and wants the AWS architecture
- The user asks for a tech stack recommendation
- The user asks for an architecture diagram of a system that doesn't exist yet (greenfield design)
- The user wants to feed the output into `build-resources` for IaC generation

**Do NOT use it when:**
- The user already has IaC and wants it modified (use `build-resources` directly)
- The user wants a code review or conceptual explanation
- The system is non-AWS and the user has not opted into AWS (ask first)
- The user only wants a diagram of an existing system (use `ar-diagram` directly)

## Process (8 steps — execute in order)

1. **Parse the requirement.** Identify the workload archetype (Section A), data shape, integration points, regulatory cues.
2. **Auto-classify NFRs.** Use the tier system in Section B to assign a scale tier. Record every assumed NFR explicitly.
3. **Map workload to services.** Use the decision matrix in Section C as the starting point. Refine based on the requirement's specifics.
4. **Identify rejected alternatives.** For each major component (compute, DB, messaging, orchestration), name at least one credible alternative and why it lost.
5. **Map decisions to Well-Architected pillars.** Each major decision must explicitly serve at least one pillar. Document this in the output.
6. **Derive the output directory.** Convert the system name to `snake_case` (e.g., "Order Manager" → `order_manager`). All artifacts for this run go under `output/<use_case_name>/`. Create the directory with `mkdir -p output/<use_case_name>` before writing any file.
7. **Write `output/<use_case_name>/tech-stack.md`** following the section contract in Section D. Section names and order are fixed — `build-resources` parses them.
8. **Invoke the `ar-diagram` skill** with the component list and data flows from `tech-stack.md`. Instruct `ar-diagram` to write its output files (`<system-name>.md` and `<system-name>.drawio`) into `output/<use_case_name>/`. Both files are part of this skill's output.

After step 8, present all three files with their absolute paths under `output/<use_case_name>/`.

---

## Section A: Workload archetypes

Classify the requirement into one of these archetypes. The archetype drives compute and messaging choices more than anything else.

| Archetype | Cues in the requirement | Default compute |
|---|---|---|
| **Sync OLTP API** | "user submits", "API returns", "request/response", request-scoped latency budget | API Gateway + Lambda or ECS Fargate |
| **Async event-driven workflow** | "on event", "after X happens", multi-step orchestration, retries, long-running | Step Functions + Lambda |
| **Batch ETL** | "nightly", "process files", "aggregate", "data warehouse", "reports" | Glue / EMR / Batch |
| **Streaming** | "real-time", "continuous", "as it arrives", "millions of events" | Kinesis / MSK + Lambda or Flink on KDA |
| **Scheduled jobs** | "every N minutes", "cron", "periodic" | EventBridge Scheduler + Lambda or ECS task |
| **ML inference** | "predict", "classify", "embedding", "RAG", "LLM" | Bedrock / SageMaker endpoint |

A single system often combines archetypes (e.g., the order manager example combines Sync OLTP + Async workflow). Identify the dominant pattern per subsystem.

## Section B: NFR auto-classification

When the user does not provide non-functional requirements, infer them from the requirement's domain and apply the tier below. **Print the chosen tier and assumed values in the `## Assumptions` section of the output** — they must be visible and tunable, never silent.

| Tier | Throughput | p99 latency | Multi-AZ | Multi-region | Consistency | Typical examples |
|---|---|---|---|---|---|---|
| **S (Small)** | <10 req/s | <500 ms | optional | no | eventual OK | internal tool, prototype, admin app |
| **M (Medium)** | 10–1,000 req/s | <200 ms | required | active-passive optional | strong on hot path | trading order manager, e-commerce checkout, partner API |
| **L (Large)** | 1,000+ req/s | <100 ms | required | active-active | strong with regional fallback | consumer-scale API, payment rails, ad serving |
| **B (Batch)** | throughput-bound | latency irrelevant | required for state stores | no | eventual | nightly ETL, report generation, document processing |
| **Stream** | events/sec specified by source | <1 s end-to-end | required | depends on source | at-least-once default | log analytics, fraud detection, telemetry |

**Domain heuristics** (override generic tier when these apply):
- Anything financial / trading / payments → at minimum Tier M, strong consistency, multi-AZ, audit trail with S3 Object Lock
- Healthcare / PII → HIPAA controls (KMS CMK, no public buckets, VPC endpoints, BAA-eligible services only)
- Public-facing consumer → WAF + CloudFront + rate limiting
- Internal/employee-only → Cognito or SSO via IAM Identity Center, no public endpoints

**RPO/RTO defaults if unspecified:**
- Tier S: RPO 24h / RTO 4h
- Tier M: RPO 1h / RTO 30min
- Tier L: RPO <5min / RTO <5min
- Tier B: RPO = batch interval / RTO = batch interval

## Section C: AWS decision matrix (starting point)

This is a starting matrix, not gospel. Override when the requirement justifies it — and document the override.

**Compute**
- Sync OLTP, spiky load, <15min execution → **Lambda**
- Sync OLTP, steady load, long-running connections (e.g., WebSocket, SSE) → **ECS Fargate**
- Heavy CPU/GPU, custom AMIs, sustained load → **EC2** or **ECS on EC2**
- Container orchestration with K8s familiarity → **EKS** (only if team has the operational maturity; default to ECS otherwise)
- Async orchestration → **Step Functions** (Express for high-volume <5min, Standard for long-running durable)
- Batch jobs → **AWS Batch** for compute-heavy, **Glue** for Spark ETL

**Storage / databases**
- Relational, ACID, complex joins, reporting → **Aurora PostgreSQL** (or Aurora MySQL)
- Key-value or document, single-digit ms reads, predictable access patterns → **DynamoDB**
- Cache, sub-ms hot reads → **ElastiCache (Valkey)** — prefer Valkey over Redis OSS for new builds
- Object storage, blobs, audit, archives → **S3** (with Object Lock for compliance)
- Time-series / metrics → **Timestream** or **CloudWatch Metrics**
- Search → **OpenSearch Serverless**
- Analytics warehouse → **Redshift Serverless** or **Athena** over S3 (Athena for ad-hoc, Redshift for sustained BI)
- Vector → **OpenSearch**, **Aurora pgvector**, or **Bedrock Knowledge Bases**

**Messaging / events**
- Point-to-point queue with FIFO ordering → **SQS FIFO** (use `MessageGroupId` for partition ordering)
- Point-to-point, no ordering → **SQS Standard**
- Pub/sub event routing → **EventBridge** (custom event bus for domain events)
- Streaming, replay, fan-out → **Kinesis Data Streams** or **MSK**
- Push notifications → **SNS**

**Orchestration**
- Multi-step workflow, branching, retries → **Step Functions**
- DAG-based data pipelines → **MWAA (Managed Airflow)**

**Data engineering**
- Spark / serverless ETL → **Glue 4+**
- Complex Spark with custom tuning → **EMR on EKS** or **EMR Serverless**
- Ad-hoc SQL on S3 → **Athena**

**API / edge**
- REST → **API Gateway HTTP API** (cheaper, faster than REST API; choose REST API only if you need WAF, request validation, API keys at the edge)
- GraphQL → **AppSync**
- WebSocket → **API Gateway WebSocket** or AppSync subscriptions
- CDN → **CloudFront**
- DDoS / bot → **WAF + Shield Advanced** (latter only for Tier L)

**Security defaults (every spec)**
- KMS CMK for at-rest encryption; never default `aws/...` keys for sensitive data
- Secrets Manager for credentials, NEVER environment variables
- VPC endpoints for S3, DynamoDB, Secrets Manager (avoid NAT Gateway egress costs and exposure)
- IAM least-privilege per function/task (no `*:*` policies)
- CloudTrail org-wide
- Config rules for compliance posture

**Observability defaults**
- CloudWatch metrics with custom dimensions
- Structured JSON logs (one schema across services)
- X-Ray tracing across orchestrations
- At least 3 SLI alarms per critical path

---

## Section D: Output 1 — `tech-stack.md` section contract

The `build-resources` skill parses this file by H2 section name. **Section names, order, and presence of all 15 sections are mandatory.** Use canonical AWS service names.

```markdown
# Technical Specification: <System Name>

## Assumptions
- **NFR Tier**: <S | M | L | B | Stream>
- **Throughput**: <assumed value>
- **Latency budget (p99)**: <assumed value>
- **Consistency**: <strong | eventual>
- **RPO / RTO**: <assumed values>
- **Compliance**: <none | SOX | HIPAA | PCI | GDPR | ...>
- **Region strategy**: <single-region multi-AZ | active-passive | active-active>
- **Other assumptions**: <list>

> If any assumption is wrong for your context, fix it here and regenerate downstream artifacts.

## Workload Archetype
<one of: Sync OLTP API | Async event-driven | Batch ETL | Streaming | Scheduled | ML inference | Hybrid>
<one paragraph explaining which subsystems map to which archetype>

## Language & Runtime
- **Language**: <Python 3.12 | Java 21 | Go 1.22 | TypeScript/Node 20 | ...>
- **Rationale**: <why>
- **Well-Architected pillars served**: <Operational Excellence, Performance Efficiency, ...>

## Compute
- **<Subsystem name>**: <service + sizing>
  - Rationale: <why>
- **<Subsystem name>**: <service + sizing>
  - Rationale: <why>
- **Well-Architected pillars served**: <list>

## Concurrency & Communication
- **Sync paths**: <list with protocols>
- **Async paths**: <list with delivery semantics — at-least-once, exactly-once, FIFO ordering key, etc.>
- **Idempotency strategy**: <how duplicate events are handled>

## Storage
- **Operational DB**: <service + reason>
- **Cache**: <service + reason or N/A>
- **Object storage**: <service + reason or N/A>
- **Search / vector / time-series**: <service + reason or N/A>
- **Access patterns covered**: <bullet list>

## Messaging & Events
- **Queues**: <service + ordering/delivery guarantees>
- **Event bus**: <service + event names>
- **Streaming**: <service or N/A>

## Orchestration
- **Workflow engine**: <Step Functions Express | Standard | MWAA | N/A>
- **Critical workflows**: <list>

## Data Engineering
<If batch/streaming/analytics: pipeline stages, services, schedule. Otherwise: "Not applicable for this workload.">

## API Layer
- **Edge**: <CloudFront | none>
- **Gateway**: <API Gateway HTTP | REST | AppSync | ALB>
- **Auth**: <Cognito | IAM | JWT authorizer | mTLS | ...>
- **Rate limiting**: <strategy>

## Security & IAM
- **Encryption at rest**: <KMS CMK strategy>
- **Encryption in transit**: <TLS posture>
- **Secrets**: <Secrets Manager scope>
- **Network**: <VPC topology, subnets, endpoints>
- **IAM**: <role boundaries, least-privilege notes>
- **Audit**: <CloudTrail, S3 Object Lock, etc.>

## Observability
- **Metrics**: <list of key SLIs>
- **Logs**: <log groups, retention, schema>
- **Traces**: <X-Ray scope>
- **Alarms**: <list with thresholds>
- **Dashboards**: <which CloudWatch dashboards to create>

## Estimated Cost Profile
<order-of-magnitude monthly estimate with itemization at the assumed scale; flag the top 3 cost drivers>

## Rejected Alternatives
- **Compute alternative**: <service> — Rejected because <reason>
- **Database alternative**: <service> — Rejected because <reason>
- **Messaging alternative**: <service> — Rejected because <reason>
- **<Optional additional rejections>**

## Well-Architected Summary
- **Operational Excellence**: <how this design serves it>
- **Security**: <how this design serves it>
- **Reliability**: <how this design serves it>
- **Performance Efficiency**: <how this design serves it>
- **Cost Optimization**: <how this design serves it>
- **Sustainability**: <how this design serves it>
```

## Output 2 — Architecture diagram (via `ar-diagram` skill)

After writing `tech-stack.md`, invoke the `ar-diagram` skill with:

- A C4-Container-level description of the system
- The component list extracted from `tech-stack.md` Sections "Compute" through "Data Engineering"
- The data flows extracted from "Concurrency & Communication"
- Sync edges = solid; async edges = dashed; every edge labeled with protocol or event name
- Trust boundaries: VPC, account, public/private subnets — drawn explicitly
- **Output path**: `output/<use_case_name>/` — pass this path explicitly so `ar-diagram` writes files there

The `ar-diagram` skill produces:
- `output/<use_case_name>/<system-name>.md` — Mermaid + notes
- `output/<use_case_name>/<system-name>.drawio` — DrawIO XML with `mxgraph.aws4.*` icons

Both files are part of this skill's deliverable.

## Integration contract with `build-resources`

`build-resources` consumes `tech-stack.md` and parses by H2 section name. To not break the contract:

1. Section names in **Section D** above are immutable. Do not rename, reorder, or omit sections.
2. Service names must be **canonical AWS names**: "Aurora PostgreSQL" not "Postgres"; "ECS Fargate" not "Fargate containers"; "Step Functions Express" not "Step Functions (express mode)".
3. Where multiple services serve the same role (e.g., several Lambda functions), list each on its own bullet under the relevant section.
4. Cost numbers are order-of-magnitude. Do not invent precision.
5. The architecture diagram filename must match the system name (snake_case) used in `tech-stack.md`'s H1.

If a section genuinely does not apply (e.g., no data engineering for a pure OLTP system), the section header still appears with the body `Not applicable for this workload.` Do not delete the header — the parser depends on its presence.

---

## Worked example — Order Manager (full reference)

A complete worked example lives at `references/order-manager/`:
- `requirement.md` — the input
- `tech-stack.md` — Output 1
- `architecture.md` — Output 2 (Mermaid)
- `architecture.drawio` — Output 2 (DrawIO XML)

Read these files when:
- Generating a tech spec for a workload similar in shape (financial OLTP, transactional OLTP with an async settlement leg, multi-validator fan-out)
- Verifying section-contract compliance
- Sanity-checking your own output against the ground-truth structure

Additional reference patterns to add over time (not yet implemented — flag to the user if their workload matches and the reference is missing):
- `references/log-analytics-pipeline/` — streaming + batch hybrid (Kinesis → S3 → Glue → Athena)
- `references/document-processing/` — async file processing with Step Functions, S3, Textract
- `references/ml-rag-inference/` — RAG inference with Bedrock + OpenSearch + Lambda

## Anti-patterns (reject your own draft if any of these apply)

- **Hand-wavy compute pick** — "Lambda or ECS, depending on load" is not a recommendation. Pick one and justify.
- **Missing rejected alternatives** — without explicit tradeoffs, the output is overconfident slop.
- **Silent NFR assumptions** — every assumed value must be in the `## Assumptions` section.
- **Generic Well-Architected lip service** — "this design is secure" is not a Well-Architected analysis. Tie each pillar to a concrete decision in the spec.
- **Service salad** — using 15 AWS services when 6 would do. Default to fewer services; add services only when they earn their place.
- **Mixed C4 levels in the diagram** — the diagram should be Container-level: deployable units and how they communicate, not individual functions and not "AWS" as a single box.
- **Cost section that's just "depends on usage"** — give an order-of-magnitude with the top 3 drivers, even if the total ranges 10x.
- **Skipping the diagram** — the diagram is not optional. If `ar-diagram` is unavailable, generate at least the Mermaid inline so `build-resources` has something to work from.

## Trigger phrasing examples

**Trigger:**
- "Design an order manager for our trading system."
- "What AWS architecture should I use for a document-processing pipeline?"
- "Architect a backend for an internal expense-reporting tool."
- "I need a tech spec for a real-time fraud-detection pipeline."
- "How would you build a RAG inference service on AWS?"

**Do not trigger:**
- "What's the difference between Aurora and DynamoDB?" (Comparative explanation, not a system design.)
- "Review my Terraform for a Lambda function." (Code review, not greenfield design.)
- "Explain the AWS Well-Architected Framework." (Conceptual.)
