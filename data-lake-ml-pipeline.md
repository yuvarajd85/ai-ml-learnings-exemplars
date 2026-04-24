# Data Lake + ML Pipeline Architecture

C4 Container-level data-flow view: Nasdaq and 3rd-party vendor API ingestion → S3 Raw → Glue curation → S3 Curated (Iceberg) → Bedrock LLM employee extraction → SageMaker XGBoost/CatBoost training and real-time inference endpoint.

## Diagram

```mermaid
flowchart LR

  subgraph EXT ["External Systems"]
    nasdaq["Nasdaq&#xa;Vendor API"]
    vendor2["3rd Party&#xa;Vendor 2 API"]
  end

  subgraph AWS ["AWS Account"]

    subgraph INGEST ["Ingestion — Step Functions + Lambda"]
      sfn["Step Functions&#xa;(Scheduler)"]
      ing["Lambda&#xa;(API Connector)"]
    end

    subgraph RAW ["S3 — Raw Layer  ·  native text / JSON"]
      s3raw[("S3 Raw Bucket&#xa;/raw/nasdaq/  ·  /raw/vendor2/")]
    end

    subgraph CURATION ["Curation — AWS Glue"]
      glue["Glue ETL Job&#xa;(curate + aggregate)"]
    end

    subgraph CURATED ["S3 — Curated Layer  ·  Apache Iceberg"]
      s3cur[("Curated&#xa;Iceberg Tables")]
      empTbl[("Employee Details&#xa;Table (Iceberg)")]
    end

    subgraph LLMEXT ["LLM Extraction — Amazon Bedrock"]
      lamExt["Lambda&#xa;(Extraction Job)"]
      bedrock{{"Bedrock&#xa;(Claude)"}}
    end

    subgraph ML ["ML Platform — Amazon SageMaker"]
      smTrain["SageMaker Training&#xa;XGBoost / CatBoost"]
      smReg["Model&#xa;Registry"]
      smEP(["SageMaker Endpoint&#xa;(Inference API)"])
    end

  end

  nasdaq  -->|"REST API (HTTPS)"| ing
  vendor2 -->|"REST API (HTTPS)"| ing
  sfn     -.->|"schedule (cron)"| ing
  ing     -->|"write raw text"| s3raw

  s3raw   -->|"read raw records"| glue
  glue    -->|"write Iceberg"| s3cur

  s3cur   -->|"text blob rows"| lamExt
  lamExt  -->|"InvokeModel (prompt)"| bedrock
  bedrock -->|"structured employee JSON"| empTbl

  s3cur   -->|"curated features"| smTrain
  empTbl  -->|"employee features"| smTrain
  smTrain -->|"register"| smReg
  smReg   -->|"deploy"| smEP
```

## Notes

- **Scope**: Two external vendor API sources; Lambda-based pull ingestion scheduled by Step Functions; two-tier S3 data lake (raw text → curated Apache Iceberg); Bedrock Claude LLM extraction from a text-blob column into a structured employee table; SageMaker XGBoost/CatBoost classification + rating model training, model registry, and a real-time inference endpoint.
- **Deliberate omissions**: Glue Data Catalog table registration, S3 event notifications (alternative ingest trigger), IAM roles and KMS encryption, CloudWatch alarms and SageMaker Model Monitor, retraining pipeline, and Feature Store — ask for an ops/failure-path view to add these.
- **Assumptions**: Ingestion is pull-based (Lambda scheduled via Step Functions cron); Glue writes Parquet-backed Iceberg tables registered in the Glue Catalog; LLM extraction runs as a separate batch Lambda (not inline with curation); SageMaker training reads directly from S3 (no Feature Store assumed — verify if feature reuse across models is needed).
- **DrawIO file**: open `data-lake-ml-pipeline.drawio` in [app.diagrams.net](https://app.diagrams.net) for the full AWS-icon version.
