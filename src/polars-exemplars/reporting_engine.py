from polars import DataFrame, Decimal
import polars as pl
from datetime import datetime as dt
import os

def main():
    spec = {
        "cols" : ["Tran_date","Tran_Ref_No","Location","Amount"],
        "transformations" : {"Tran_date":"convert_to_date","Amount":"convert_to_decimal","Merchant":"convert_to_string"},
        "filter_specs" : [{"field" : "Amount","operation" : ">","value":"-10"}],
        "output_format" : "csv"
    }

    transformation_function_dict = {
        "convert_to_string": convert_to_string,
        "convert_to_date" : convert_to_date,
        "convert_to_decimal": convert_to_decimal
    }

    filter_function_dict = {
        ">" : filter_greater_than,
        ">=" : filter_greater_than_equal_to,
        "gt" : filter_greater_than,
        "<": filter_lesser_than,
        "<=": filter_lesser_than_equal_to,
        "lt": filter_lesser_than,
    }

    df = pl.read_csv(f"C://Users//{os.getlogin()}//Desktop//TxnIn.txt",infer_schema_length=100000,ignore_errors=True)
    print(df.head())

    #apply column filters
    df = df.select(spec.get("cols"))

    #apply column transformation
    for key, val in spec.get("transformations").items():
        if key in spec.get("cols"):
            df = transformation_function_dict.get(val).__call__(df,key)

    print(df.head())

    df = df.with_columns([
        pl.lit(pl.col("Amount") * -1.0 ).alias("Amount")
    ])

    for filter_spec in spec.get("filter_specs"):
        operation = filter_spec.get("operation")
        field = filter_spec.get("field")
        value = filter_spec.get("value")

        df = filter_function_dict.get(operation).__call__(df,field,value)

    print(df.head())

def convert_to_string(df:DataFrame, field:str):
    return df.with_columns([
        pl.col(field).cast(pl.Utf8).alias(field)
    ])

def convert_to_date(df:DataFrame, field:str):
    return df.with_columns([
        pl.col(field).str.to_date("%m/%d/%Y").alias(field)
    ])

def convert_to_decimal(df:DataFrame, field:str):
    return df.with_columns([
        pl.col(field).cast(pl.Decimal(scale=5)).alias(field)
    ])

def filter_greater_than(df:DataFrame,field:str,value:str):
    print(df.select(field).dtypes[0])
    match df.select(field).dtypes[0]:
        case "Decimal(precision=38, scale=5)" :
            value = Decimal(value)
        case _:
            value

    return df.filter(pl.col(field) > value)

def filter_greater_than_equal_to(df:DataFrame,field:str,value:str):
    print(df.select(field).dtypes)
    return df.filter(pl.col(field) >= value)

def filter_lesser_than(df:DataFrame,field:str,value:str):
    print(df.select(field).dtypes[0])
    match df.select(field).dtypes[0]:
        case "Decimal(precision=38, scale=5)" :
            value = Decimal(value)
        case _:
            value

    return df.filter(pl.col(field) < value)

def filter_lesser_than_equal_to(df:DataFrame,field:str,value:str):
    print(df.select(field).dtypes)
    return df.filter(pl.col(field) <= value)


if __name__ == '__main__':
    main()