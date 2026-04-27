# Order Manager — Architecture

C4 Container-level view of the order manager system. Shows deployable units, communication paths, trust boundaries, and external integrations.

## Diagram

```mermaid
flowchart LR
  Client([Client / Trader UI])
  APIG[API Gateway<br/>HTTP API + JWT]

  subgraph VPC[VPC — us-east-1, multi-AZ]
    direction TB
    subgraph AppTier[Private subnet — app tier]
      Intake[Order Intake<br/>Lambda]
      SfnVal[Validation Workflow<br/>Step Functions Express]
      Validators[Validators<br/>4 × Lambda Parallel]
      Submitter[Order Submitter<br/>Lambda]
      SfnSettle[Settlement Workflow<br/>Step Functions Standard]
      SettleFn[Settlement Processor<br/>Lambda]
    end
    subgraph DataTier[Private subnet — data tier]
      Cache[(ElastiCache<br/>Valkey)]
      Aurora[(Aurora PostgreSQL<br/>multi-AZ)]
    end
  end

  SQS{{SQS FIFO<br/>order_submission_queue}}
  EB{{EventBridge<br/>order-events bus}}
  S3[(S3 Audit Bucket<br/>Object Lock)]
  Broker[Broker / Exchange<br/>Gateway]

  Client -->|HTTPS POST /orders| APIG
  APIG -->|invoke| Intake
  Intake -->|StartSyncExecution| SfnVal
  SfnVal -->|Parallel state| Validators
  Validators -->|GET account / position| Cache
  Validators -->|SQL on cache miss| Aurora
  SfnVal -->|SendMessage MessageGroupId=account_id| SQS
  SQS -->|invoke| Submitter
  Submitter -->|submit order| Broker
  Submitter -->|INSERT pending_orders| Aurora
  Submitter -.audit object.-> S3
  Submitter -.emit order.submitted.-> EB
  Broker -.settlement event.-> EB
  EB -->|rule → start execution| SfnSettle
  SfnSettle -->|invoke| SettleFn
  SettleFn -->|BEGIN; INSERT history; DELETE pending; COMMIT| Aurora
  SettleFn -.audit object.-> S3
  SettleFn -.emit order.settled.-> EB

  classDef ext fill:#e1d5e7,stroke:#9673a6,color:#000
  class Broker ext
```

## Notes

- **Scope**: Sync intake path (Client → API Gateway → validation → SQS) and async settlement path (Broker settlement event → EventBridge → Step Functions Standard → atomic Aurora move). Trust boundaries shown: VPC, app-tier subnet, data-tier subnet. External actors: Client, Broker / Exchange Gateway.
- **Deliberate omissions**:
  - Cross-cutting services (KMS, Secrets Manager, CloudWatch, X-Ray, CloudTrail) — referenced in `tech-stack.md` under Security & Observability; including them here would obscure the data flow.
  - DLQ on the FIFO queue and Step Functions error states — implicit; called out in `tech-stack.md`.
  - The IdP (Cognito or external OIDC) and the JWT authorizer detail — abstracted into the API Gateway label.
  - The 4 individual validator Lambdas are collapsed into a single `Validators` node since they share a pattern; the workflow detail is in `tech-stack.md`.
- **Assumptions**: See `tech-stack.md` `## Assumptions`. Tier M scale, single-region multi-AZ, financial-domain compliance posture.
- **DrawIO file**: open `architecture.drawio` in [app.diagrams.net](https://app.diagrams.net) for the AWS-icon view with proper service iconography.
