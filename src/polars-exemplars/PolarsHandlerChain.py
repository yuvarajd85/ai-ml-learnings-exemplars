'''
Created on 4/29/2025 at 10:21 PM
By yuvaraj
Module Name: PolarsHandlerChain
'''
from dotenv import load_dotenv

from database_service.dbservice import PGDB_Service_Impl
from handlers.user_handler import User
import polars as pl
from polars import DataFrame

load_dotenv()

handler_chain = [User]

def main():
    db_service = PGDB_Service_Impl()

    data = [handler(db_service).get_data() for handler in handler_chain]

    for datum in data:
        df = pl.DataFrame(datum)
        print(df.head())




if __name__ == '__main__':
    main()
