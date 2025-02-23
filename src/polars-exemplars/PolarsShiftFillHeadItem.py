'''
Created on 2/22/2025 at 9:12 PM
By yuvaraj
Module Name: PolarsShiftFillHeadItem
'''
import polars as pl
from dotenv import load_dotenv
from polars import DataFrame

load_dotenv()

def main():
    pl.Config.set_tbl_cols(200)
    pl.Config.set_tbl_rows(200)
    df: DataFrame = pl.read_csv(f"../resources/datasets/sales_data.csv")
    df = df.with_columns([
        pl.col("Sale_Date").str.to_date().alias("Sale_Date")
    ]).with_columns([
        pl.col("Sale_Date").dt.year().alias("Sale_Year"),
        pl.col("Sale_Date").dt.month().alias("Sale_Month"),
        pl.col("Sale_Date").dt.day().alias("Sale_Day"),

    ]).sort(["Sale_Date","Sales_Rep"])
    print(df.head(20))

    """
    1. Following step will shift the `Unit_Price` column down by one record within each 
        group["Sale_Year","Sale_Month","Sale_Day","Sales_Rep"] and the first record will be left null, eventually creating 
        a new column named `Prev_Unit_Price`. 
    2. First record each group with the null value is then filled with the first record value of `Unit_Price` column of each group.  
    """

    df = df.with_columns([
        pl.col("Unit_Price").shift(n=1).over(["Sale_Year","Sale_Month","Sale_Day","Sales_Rep"]).fill_null(pl.col("Unit_Price").first().over(["Sale_Year","Sale_Month","Sale_Day","Sales_Rep"])).alias("Prev_Unit_Price")
    ])


    print(df.head(20))
    
if __name__ == '__main__':
    main()