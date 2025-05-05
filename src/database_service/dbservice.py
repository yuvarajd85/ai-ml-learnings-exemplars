'''
Created on 4/29/2025 at 10:31 PM
By yuvaraj
Module Name: dbservice
'''
import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()


class PGDB_Service_Impl():
    def __init__(self):
        self.db_prop = {
                            "database" : os.getenv("dbname"),
                            "host" : os.getenv("dbhost"),
                            "user" : os.getenv("dbuser"),
                            "password" : os.getenv("dbcred")
                        }
        self.conn = psycopg2.connect(**self.db_prop)

    def get_connection(self):
        return self.conn

    def execute_fetch_query(self, query:str):
        cursor = self.conn.cursor()
        cursor.execute(query=query)
        return cursor.fetchall()
