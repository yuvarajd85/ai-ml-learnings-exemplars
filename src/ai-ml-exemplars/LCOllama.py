'''
Created on 3/2/26 at 12:02 PM
By yuvarajdurairaj
Module Name LCOllama
'''
from Prompts import TAX_SYSTEM_PROMPT
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def main():
    llm_model = ChatOllama(
        model="llama3.2",
        temperature=0.2,
        top_p=0.2,
        max_tokens=4096
    )
    prompt = f"Calculate tax for the marrired joint filing with a total income of $360,000 with two kids and with two houses with mortgage interest payment of $31,000. This is for filing year 2025. State of residence is Pennsylvania"

    messages = [
        SystemMessage(content=TAX_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    response = llm_model.invoke(messages)

    print(response)


if __name__ == '__main__':
    main()
