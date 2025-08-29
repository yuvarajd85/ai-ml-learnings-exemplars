'''
Created on 8/28/2025 at 9:43 PM
By yuvaraj
Module Name: PineconeDeleteIndex
'''
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()


def main():
    pc = Pinecone(
        api_key=os.getenv('PINECONE_API_KEY')
    )

    pinecone_index_name='sagemaker-guide-embeddings'
    if pinecone_index_name in pc.list_indexes().names():
        pc.delete_index(name=pinecone_index_name)

        print(f"Pinecone Index {pinecone_index_name} deleted")
    else:
        print(f"Pinecone index {pinecone_index_name} does not exist or already deleted")


if __name__ == '__main__':
    main()
