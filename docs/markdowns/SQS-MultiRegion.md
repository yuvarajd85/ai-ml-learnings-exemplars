```mermaid
sequence
AWS DynamoDB
DynamoDB Region 1 ->> Lambda Region 1
DynamoDB Region 2 ->> Lambda Region 2
DynamoDB Region 3 ->> Lambda Region 3
Lambda Region 1 ->> DynamoDB Region 1
Lambda Region 2 ->> DynamoDB Region 2
Lambda Region 3 ->> DynamoDB Region 3
```