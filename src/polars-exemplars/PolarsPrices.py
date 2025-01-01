import polars as pl
from polars import DataFrame


def main():
    pl.Config.set_tbl_cols(200)
    pl.Config.set_tbl_rows(2000)
    df : DataFrame = pl.read_csv("E://Practice-Datasets//archive//prices-split-adjusted.csv")
    df = df.with_columns([
        pl.col("date").str.to_date().alias("date")
    ]).sort(["symbol","date"])

    print(df.head(100))

    df = df.upsample(time_column="date",every="1d",group_by=["symbol"]).select(pl.all().forward_fill()).sort(["symbol","date"])

    print(df.head(3000))

if __name__ == "__main__":
    main()

