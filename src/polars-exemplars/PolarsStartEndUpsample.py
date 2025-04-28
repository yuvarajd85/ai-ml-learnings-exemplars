'''
Created on 4/28/2025 at 1:27 PM
By yuvaraj
Module Name: PolarsStartEndUpsample
'''
from datetime import date

from dotenv import load_dotenv
import polars as pl
from polars import DataFrame
load_dotenv()


def main():
    pl.Config.set_tbl_cols(200)
    pl.Config.set_tbl_rows(200)
    df: DataFrame = pl.read_csv("../resources/datasets/tckr-start-end-date.csv")
    df = df.with_columns([
        pl.col("start_date").str.to_date("%Y-%m-%d").alias("start_date"),
        pl.col("end_date").str.to_date("%Y-%m-%d").alias("end_date")
    ]).with_columns([
        pl.col("start_date").alias("effective_date")
    ])

    print(df.head())

    df_end = df.with_columns([
        pl.when(
            pl.col("end_date") == date(year=9999, month=12, day=31)
        ).then(
            pl.lit(date.today())
        ).otherwise(pl.col("end_date")).alias("effective_date")
    ])

    df = df.vstack(df_end).sort(by=["tckr","effective_date","code"]).upsample(time_column="effective_date",every="1d",group_by=["tckr","code"]).select(pl.all().forward_fill())

    print(df.head(200))

    df = df.pivot(on="code",index=["tckr","effective_date","start_date","end_date"],values="price")

    print(df.head(200))

if __name__ == '__main__':
    main()
