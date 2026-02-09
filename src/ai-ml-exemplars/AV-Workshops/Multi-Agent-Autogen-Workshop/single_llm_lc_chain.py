'''
Created on 2/7/26 at 9:04 AM
By yuvarajdurairaj
Module Name single_llm
'''
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    model = "gpt-4o-mini"

    llm = ChatOpenAI(
        model=model,
        temperature=0.2,
        max_tokens=4096,
        timeout=None,
        max_retries=2,
        api_key=openai_api_key
    )

    system_prompt = f"""
    You are a senior Data Scientist.
    Explain concepts with math intuition, implementation details,
    and practical tradeoffs. Avoid fluff. Be precise and technical.  
    """

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "user",
                """
                Context:
                {context}

                Question:
                {question}
                """
            ),
        ]
    )

    llm_chain = prompt_template | llm | StrOutputParser()

    response = llm_chain.invoke(
        {
            "context": "User is building a spam classifier using TF-IDF + Logistic Regression in sklearn.",
            "question": "Explain logistic regression with TF-IDF features and how optimization works."
        }
    )

    print(response)


if __name__ == '__main__':
    main()
