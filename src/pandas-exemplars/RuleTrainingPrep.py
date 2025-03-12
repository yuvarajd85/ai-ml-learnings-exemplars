'''
Created on 2/28/2025 at 12:36 AM
By yuvaraj
Module Name: RuleTrainingPrep
'''
from dotenv import load_dotenv
import polars as pl
import random

from numpy.random.mtrand import operator

load_dotenv()


def main():
    columns = ["age", "salary", "holdings", "ticker_symbol", "height", "weight","cusip"]
    functions = ["ColumnValue","ColumnLength", "ColumnDataType"]
    operators = [">", "<", "=", ">=", "<="]
    datatypes = ["int", "string", "float", "boolean"]

    train_data = []

    for _ in range(10000):
        column = random.choice(columns)
        function = random.choice(functions)
        operator = random.choice(operators)
        datatype = random.choice(datatypes)

        match function:
            case "ColumnValue":
                value = random.randint(1,1000000)
                input_expr = f"{column} {operator} {value}"
                output_expr = f"{function} `{column}` {operator} {value}"
            case "ColumnLength":
                value = random.randint(1,10000)
                input_expr = f"length of {column} {operator} {value}"
                output_expr = f"{function} `{column}` {operator} {value}"
            case "ColumnDataType":
                input_expr = f"type of {column} is {datatype}"
                output_expr = f"{function} `{column}` = {value}"
            case _:
                value = random.randint(1, 1000000)
                input_expr = f"{column} {operator} {value}"
                output_expr = f"{function} `{column}` {operator} {value}"

        train_data.append({"input" : input_expr,"output": output_expr})

    df = pl.DataFrame(train_data)
    print(df.head())

    df.write_csv(f"../resources/datasets/rule_train_dataset.csv")

if __name__ == '__main__':
    main()
