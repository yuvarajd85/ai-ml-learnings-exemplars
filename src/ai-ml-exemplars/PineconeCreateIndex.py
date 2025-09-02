'''
Created on 8/28/2025 at 5:15 PM
By yuvaraj
Module Name: PineconeCreateIndex
'''
import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


def main():
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

    print(pc.list_indexes().names())
    pinecone_index_name = 'rag-text-embedding'

    if pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(name=pinecone_index_name,
                        dimension=3072,
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )
        while not pc.describe_index(pinecone_index_name).index.status['ready']:
            time.sleep(1)

        print("Pinecone Index provisioned")
    else:
        print("Pinecone Index Already Provisioned")



if __name__ == '__main__':
    main()
