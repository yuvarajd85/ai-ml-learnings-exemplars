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

from sympy.physics.units import temperature

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

    join_query_sepc = {
	"base_table" : {
			"table" : "db.table_emp",
			"cols" : ["name","id"],
			"filters" : [{"column" : "id", "operator": "IN", "value": "1,23,5"}]
		},
	"join_tables" : [
		{
			"table" : "db.table_emp",
			"cols" : ["id","dept","division"],
			"how" : "inner",
			"leftTable" : "db.table_dept",
			"leftKeys" : ["id"],
			"rightKeys" : ["id"],
			"filters" : [{"column" : "dept", "operator": "IN", "value": "assembly,painting"}]
		},
		{
			"table" : "db.table_emp",
			"cols" : ["id","service_start_date","service_end_date","service_dept"],
			"how" : "inner",
			"leftTable" : "db.table_service",
			"leftKeys" : ["id"],
			"rightKeys" : ["id"],
			"filters" : [{"column" : "service_start_date", "operator": "<", "value": ":current-date"}]
		}
	]
}
    query_spec_str = json.dumps(query_spec)

    # prompt = f"Generate the sql using the following query spec: {query_spec_str}"

    prompt = """
    Generate the sql using the following query spec: 
    {
        "table" : "db.table_a",
        "cols" : ["name","id"],
        "filters" : [{"column" : "id", "operator": "IN", "value": "1,23,5"}]
    }
    """

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages" : [
            {
                "role" : "user",
                "content" : [{"type": "text", "text": prompt}]
            }
        ],
        "temperature" : 0.5
    })

    prompt_request = {
        "body" : body,
        "modelId" : "anthropic.claude-3-5-sonnet-20240620-v1:0"
    }

    bedrock_client = get_client(service_name="bedrock-runtime", access_dict=access_dict)

    # response = bedrock_client.list_foundation_models()

    response = bedrock_client.invoke_model(
        modelId=prompt_request['modelId'],
        body=prompt_request['body']
    )

    response_json = json.loads(response.get('body').read())
    response_text = response_json.get('content')[0].get('text')
    print(response_text)

def get_client(service_name : str, access_dict : Dict):
    return boto3.Session(aws_access_key_id=access_dict.get("access_key"), aws_secret_access_key=access_dict.get("secret_access_key"),region_name="us-east-1").client(service_name=service_name)


if __name__ == '__main__':
    main()
