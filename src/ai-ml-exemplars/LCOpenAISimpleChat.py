'''
Created on 1/19/26 at 11:51 AM
By yuvarajdurairaj
Module Name LCSimpleChat
'''
import os

from dotenv import load_dotenv
load_dotenv()
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def main():
    openai.api_key = os.environ['OPENAI_API_KEY']
    chatgpt_model = "gpt-4o"
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a smart and intelligent bot interface with Mastery skill on all the subjects in the world"),
        ("human", "{question}")
    ])

    llm_model_obj = ChatOpenAI(
        model=chatgpt_model,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.2
    )

    chat_chain = prompt | llm_model_obj

    response = chat_chain.invoke({
        "question" : "Explain about coreless motor"
    })

    print(response.content)


if __name__ == '__main__':
    main()
