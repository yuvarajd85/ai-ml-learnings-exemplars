'''
Created on 2/7/26 at 9:40 AM
By yuvarajdurairaj
Module Name single_llm
'''
from openai import OpenAI
from constants import openai_api_key

client = OpenAI(api_key=openai_api_key)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
