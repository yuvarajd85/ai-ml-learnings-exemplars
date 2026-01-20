'''
Created on 1/19/26 at 12:51 PM
By yuvarajdurairaj
Module Name LCGeminiSimpleChat
'''
import os

from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    gemini_model = "gemini-2.5-flash-lite"
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a smart and intelligent bot interface with Mastery skill on all the subjects in the world"),
        ("human", "{question}")
    ])

    llm_model_obj = ChatGoogleGenerativeAI(
        model=gemini_model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2
    )

    chat_chain = prompt | llm_model_obj

    response = chat_chain.invoke({
        "question": "Explain Langchain library for google genai"
    })

    print(response.content)


if __name__ == '__main__':
    main()
