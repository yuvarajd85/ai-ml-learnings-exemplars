# Insurance Claim Intake & Adjuster Workspace — Container Architecture

C4 Container-level view of all deployable units, trust boundaries, and primary data flows.
Solid edges = sync; dashed edges = async / batch.

## Diagram

```mermaid
flowchart LR
  Customer(["Customer\nweb / mobile"])
  Adjuster(["Adjuster\nbrowser"])
  OpsMgr(["Ops Manager\nbrowser"])
  CorpIdP(["Corporate IdP\nSAML"])

  subgraph PublicEdge ["Public Edge"]
    CF["CloudFront + WAF"]
    CogCust["Cognito\nCustomer Pool"]
    CogAdj["Cognito\nAdjuster Pool\n(SAML fed.)"]
  end

  subgraph VPC ["AWS VPC — us-east-1"]
    subgraph Compute ["Private subnet — compute"]
      IntakeLambda["Lambda\nclaim-intake-handler"]
      AdjLambda["Lambda\nadjuster-api-handler"]
      FinanceLambda["Lambda\nfinance-report-gen"]
    end
    subgraph DataSub ["Private subnet — data"]
      Valkey[("ElastiCache\nValkey")]
      DDB[("DynamoDB\nidempotency")]
    end
    subgraph IsoSub ["Isolated subnet — database"]
      Aurora[("Aurora PostgreSQL 16\nMulti-AZ + 2 read replicas")]
      OpenSearch[("OpenSearch Serverless\ninsurance-claims")]
    end
  end

  subgraph AsyncPipeline ["AWS Managed Services — via VPC Endpoints"]
    subgraph APILayer ["API Layer"]
      APIGWC["API Gateway HTTP\nclaims-customer-api"]
      APIGWA["API Gateway HTTP\nclaims-adjuster-api"]
    end
    subgraph IngestionFlow ["Document Ingestion"]
      SQS{{"SQS\ndocument-intake-queue"}}
      IngSFN["Step Functions\nDocumentIngestionWorkflow"]
      Textract["AWS Textract\nAnalyzeDocument"]
      Bedrock["Bedrock Runtime\nClaude Haiku"]
    end
    subgraph TriageFlow ["Triage & Routing"]
      TriSFN["Step Functions\nTriageAndRoutingWorkflow"]
    end
    subgraph MessagingLayer ["Messaging & Notifications"]
      EB{{"EventBridge\ninsurance-domain-events"}}
      SNS{{"SNS\nclaim-status-notifications"}}
      SES["Amazon SES"]
      Push["SNS Mobile Push\nAPNs / FCM"]
    end
    subgraph StorageLayer ["Object Storage"]
      S3Docs[("S3 claims-documents\nObject Lock Compliance 7yr")]
      S3Audit[("S3 claims-audit-log\nObject Lock Compliance 7yr")]
      S3Analytics[("S3 claims-analytics\nParquet")]
    end
    subgraph AnalyticsLayer ["Analytics & BI"]
      EBSched["EventBridge Scheduler"]
      Glue["AWS Glue 4.0\nSpark ETL"]
      Redshift[("Redshift Serverless")]
      QS["Amazon QuickSight"]
      Athena["Amazon Athena"]
    end
    subgraph ComplianceLayer ["Compliance & Observability"]
      CloudTrail["AWS CloudTrail"]
      CW["Amazon CloudWatch"]
    end
  end

  subgraph ExtSys ["External Systems"]
    CorpIdP
    APNsFCM(["iOS APNs\nAndroid FCM"])
  end

  %% ── Submission flow (sync) ────────────────────────────────────────────────
  Customer -->|"HTTPS"| CF
  CF -->|"HTTPS / JWT"| APIGWC
  CogCust -->|"JWT"| APIGWC
  APIGWC -->|"invoke"| IntakeLambda
  IntakeLambda -->|"INSERT claim SQL"| Aurora
  IntakeLambda -->|"policy lookup"| Valkey
  IntakeLambda -->|"pre-signed PUT URL"| Customer
  Customer -.->|"multipart PUT"| S3Docs

  %% ── Document ingestion (async) ────────────────────────────────────────────
  S3Docs -.->|"S3 event notification"| SQS
  SQS -.->|"trigger workflow"| IngSFN
  IngSFN -.->|"AnalyzeDocument"| Textract
  IngSFN -.->|"classify complexity"| Bedrock
  IngSFN -.->|"INSERT extraction SQL"| Aurora
  IngSFN -.->|"index document"| OpenSearch
  IngSFN -.->|"extraction.completed"| EB

  %% ── Triage & routing (async) ─────────────────────────────────────────────
  EB -.->|"extraction.completed"| TriSFN
  TriSFN -.->|"complexity scoring"| Bedrock
  TriSFN -.->|"adjuster pool query SQL"| Aurora
  TriSFN -.->|"UPDATE claim status SQL"| Aurora
  TriSFN -.->|"claim.status.changed"| EB

  %% ── Notifications (async) ────────────────────────────────────────────────
  EB -.->|"claim.status.changed"| SNS
  SNS -.->|"email fanout"| SES
  SNS -.->|"push fanout"| Push
  SES -.->|"email notification"| Customer
  Push -.->|"push notification"| APNsFCM

  %% ── Adjuster workspace (sync) ────────────────────────────────────────────
  CorpIdP -->|"SAML assertion"| CogAdj
  CogAdj -->|"JWT"| APIGWA
  Adjuster -->|"HTTPS / JWT"| APIGWA
  APIGWA -->|"invoke"| AdjLambda
  AdjLambda -->|"full-text search"| OpenSearch
  AdjLambda -->|"claim detail SQL"| Aurora
  AdjLambda -->|"metadata cache"| Valkey
  AdjLambda -->|"pre-signed GET URL"| S3Docs
  AdjLambda -->|"UPDATE status SQL"| Aurora
  AdjLambda -.->|"claim.status.changed"| EB
  AdjLambda -->|"idempotency check"| DDB

  %% ── Analytics / batch ────────────────────────────────────────────────────
  EBSched -.->|"cron 02:00 UTC"| Glue
  Glue -.->|"JDBC read"| Aurora
  Glue -.->|"Parquet write"| S3Analytics
  Glue -.->|"COPY"| Redshift
  OpsMgr -->|"HTTPS"| QS
  QS -->|"SPICE refresh"| Redshift
  EBSched -.->|"monthly trigger"| FinanceLambda
  FinanceLambda -.->|"SQL query"| Athena
  Athena -.->|"read Parquet"| S3Analytics
  FinanceLambda -.->|"CSV via SES"| SES

  %% ── Compliance / audit ───────────────────────────────────────────────────
  CloudTrail -.->|"management + data events"| S3Audit
  IntakeLambda -.->|"per-claim event record"| S3Audit
  AdjLambda -.->|"per-claim event record"| S3Audit
```

## Notes

- **Scope**: All AWS-deployed containers and AWS managed services. Primary data flows between trust boundaries. C4 Container level — each box is a separately deployable or managed service unit.
- **Deliberate omissions**: DLQ (document-intake-dlq), VPC NAT Gateways (no compute there), KMS / Secrets Manager cross-cutting connections (implied by security design), individual Lambda task functions inside Step Functions (consolidated into workflow boxes), Aurora read-replica distinction (implied by label).
- **Assumptions**: All Lambda functions execute in VPC private subnets and reach AWS managed services via VPC Interface Endpoints; OpenSearch Serverless is provisioned with VPC-based access policy; Step Functions Standard is the durable workflow engine for both orchestrations.
- **DrawIO file**: Open `insurance_claim_intake.drawio` in [app.diagrams.net](https://app.diagrams.net) for the AWS-icon version with zone containers.
