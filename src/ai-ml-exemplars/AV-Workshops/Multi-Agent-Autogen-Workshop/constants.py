'''
Created on 2/7/26 at 9:45 AM
By yuvarajdurairaj
Module Name constants
'''
import os

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
