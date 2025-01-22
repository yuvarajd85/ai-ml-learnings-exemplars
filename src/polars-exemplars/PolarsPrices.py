import polars as pl
from polars import DataFrame
import polars_xdt as xdt

def main():
    pl.Config.set_tbl_cols(200)
    pl.Config.set_tbl_rows(2000)
    df : DataFrame = pl.read_csv("E://Sample-Datasets//Practice-Datasets//archive//prices-split-adjusted.csv")

    df = df.with_columns([
        pl.col("date").str.to_date().alias("date")
    ]).sort(["symbol","date"])

    df = df.with_columns([
        ((pl.col("low") + pl.col("high")) / 2).alias("avg"),
        (pl.col("open") - pl.col("close")).alias("open_close_diff"),
        ((pl.col("open") - pl.col("close")) / pl.col("open") * 100).alias("percentage_change")
    ])

    df = df.sort(["symbol","date"]).upsample(time_column="date",every="1d",group_by=["symbol"]).select(pl.all().forward_fill()).sort(["symbol","date"])

    print(df.head(3000))

    df = df.with_columns([
        (pl.col("date").dt.weekday().is_in([6,7])).alias("non_bus_day"),
        xdt.is_workday("date").alias("bus_day")
    ])

    print(df.head(10))

    df = df.with_columns([
        pl.col("date").dt.offset_by("-1mo").dt.truncate("1mo").dt.offset_by("-1d").alias("prev_month")
    ]).with_columns([
        (pl.col("prev_month") - pl.col("date")).dt.total_days().alias("diff_in_days")
    ])

    print(df.head(20))


def dynamic_sum(row):
    if row['diff_in_days'] >= 3:
        return sum(row['low'].shift(offset).fill_null(0) for offset in range((1-row['diff_in_days']), 1) )
    else:
        return row['low']

if __name__ == "__main__":
    main()

