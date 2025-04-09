'''
Created on 4/7/2025 at 6:32 PM
By yuvaraj
Module Name: psycopg2sql
'''
import os

from dotenv import load_dotenv
import psycopg2

load_dotenv()


def main():
    db_prop = {
        "database" : os.getenv("dbname"),
        "host" : os.getenv("dbhost"),
        "user" : os.getenv("dbuser"),
        "password" : os.getenv("dbcred")
    }

    conn = psycopg2.connect(**db_prop)
    cursor = conn.cursor()

    cursor.execute("select * from users")
    rows = cursor.fetchall()

    for row in rows:
        print(row)





if __name__ == '__main__':
    main()
