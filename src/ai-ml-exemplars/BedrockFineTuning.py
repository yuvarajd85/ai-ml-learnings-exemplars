'''
Created on 2/20/2025 at 7:15 PM
By yuvaraj
Module Name: BedrockFineTuning
'''
import os
from typing import Dict

from dotenv import load_dotenv
import boto3
import json

load_dotenv()


def main():
    aws_access_key = os.getenv("aws_access_key_id")
    secret_access_key = os.getenv("aws_secret_access_key")

    print(f"acc Key: {aws_access_key} | secret key: {secret_access_key}")
    access_dict = {"access_key": aws_access_key, "secret_access_key" : secret_access_key}

    s3_client = get_client(service_name="s3",access_dict=access_dict)

    # response = s3_client.list_buckets()
    # print(response)
    query_spec = {
        "table" : "db.table_a",
        "cols" : ["name","id"],
        "filters" : [{"column" : "id", "operator": "IN", "value": "1,23,5"}]
    }

    query_spec_str = json.dumps(query_spec)

    prompt = f"Generate the sql using the following query spec: {query_spec_str}"

    bedrock_client = get_client(service_name="bedrock-runtime", access_dict=access_dict)

    # response = bedrock_client.list_foundation_models()

    response = bedrock_client.invoke_model(
        modelId="meta.llama3-3-70b-instruct-v1:0",
        contentType="string",
        accept='string',
        body=prompt
    )

    print(response)

def get_client(service_name : str, access_dict : Dict):
    return boto3.Session(aws_access_key_id=access_dict.get("access_key"), aws_secret_access_key=access_dict.get("secret_access_key"),region_name="us-east-1").client(service_name=service_name)


if __name__ == '__main__':
    main()
