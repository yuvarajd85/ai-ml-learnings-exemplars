'''
Created on 4/29/2025 at 10:22 PM
By yuvaraj
Module Name: handlers
'''
from dotenv import load_dotenv

from database_service.dbservice import PGDB_Service_Impl

load_dotenv()


class User():
    def __init__(self, db_service : PGDB_Service_Impl):
        self.db_service = db_service

    def get_data(self):
        query = f"Select * from users"
        return self.db_service.execute_fetch_query(query=query)
