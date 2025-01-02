import polars as pl
from polars import DataFrame
from bdateutil import isbday

def main():
    pl.Config.set_tbl_cols(200)
    pl.Config.set_tbl_rows(2000)
    df : DataFrame = pl.read_csv("E://Practice-Datasets//archive//prices-split-adjusted.csv")

    df = df.with_columns([
        pl.col("date").str.to_date().alias("date")
    ]).sort(["symbol","date"])

    df = df.with_columns([
        ((pl.col("low") + pl.col("high")) / 2).alias("avg"),
        (pl.col("open") - pl.col("close")).alias("open_close_diff"),
        ((pl.col("open") - pl.col("close")) / pl.col("open") * 100).alias("percentage_change")
    ])

    df = df.upsample(time_column="date",every="1d",group_by=["symbol"]).select(pl.all().forward_fill()).sort(["symbol","date"])

    print(df.head(3000))

    df = df.with_columns([
        (pl.col("date").dt.weekday().is_in([6,7])).alias("non_bus_day")
    ])

    print(df.head(10))


if __name__ == "__main__":
    main()

