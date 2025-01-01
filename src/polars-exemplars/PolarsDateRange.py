import polars as pl
import polars_xdt as pxdt

def main():
    #setting the polars configuration to print the cols to a desired number and also the rows
    pl.Config.set_tbl_cols(200)
    pl.Config.set_tbl_rows(100)

    # Create a Polars DataFrame with start and end date columns
    df = pl.DataFrame({
        "start_date": ["2023-01-01", "2023-02-01", "2023-03-01"],
        "end_date": ["2023-01-03", "2023-02-15", "2023-03-04"],
    })

    df = df.with_columns([
        pl.col("start_date").str.to_date().alias("start_date"),
        pl.col("end_date").str.to_date().alias("end_date")
    ])
    # Add a new column with date ranges as lists
    df = df.with_columns(
        pl.struct([pl.col('start_date'), pl.col('end_date')]).map_elements(
            lambda row: list(pxdt.date_range(row['start_date'], row['end_date'], "1bd", eager=True)),
            return_dtype=pl.List(pl.Date)
        ).alias("date_range")
    )

    print(df.head(10))

    df = flatten_list_to_cols(df, "date_range")

    print(df.head(10))

def flatten_list_to_cols(df, col_name):
    df = df.with_columns([
        pl.col(col_name).list.len().alias("list_len")
    ])

    max_len = df.select("list_len").max().head(1).item()

    for i in range(max_len):
        df = df.with_columns([
            pl.col(col_name).list.get(i,null_on_oob=True).alias(f"{col_name}_{i+1}")
        ])

    print(df.head(10))

    return df

if __name__ == '__main__':
    main()
