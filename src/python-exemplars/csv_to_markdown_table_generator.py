'''
Created on 1/8/2025 at 9:21 PM
By yuvaraj
Module Name csv_to_markdown_table_generator
'''
from typing import List, Tuple

from dotenv import load_dotenv
import polars as pl
from polars import DataFrame

load_dotenv()

markdown_template = f"""
<center>

### :table-name

</center>


:table

"""

def main():
    pl.Config.set_tbl_cols(200)
    file_name :str = f"../resources/datasets/wine_data.csv"
    df : DataFrame = pl.read_csv(file_name)
    print(df.head())
    row, col = df.shape

    table_name = file_name.split("/")[-1].split(".")[0]
    table = print_table(col, row, df.head(200).rows(), df.columns)

    final_template = markdown_template.replace(":table-name",table_name).replace(":table",table)

    print(final_template)

def print_table(col:int, row:int, data: List[Tuple], col_name: List):
    if col_name:
        col = len(col_name)
    if data:
        row = len(data)

    table_str = ""
    hdr = ""
    for i in range(0, (col + 1)):
        hdr+= "|"

        if not (i == col):
            if col_name:
                hdr += col_name[i]
            else:
                hdr += f"Col_{i+1}"
    table_str += f"{hdr}\n"

    sep = ""
    for i in range(0, (col + 1)):
        sep += "|"
        if not (i == col):
            if i == 0:
                sep += ":---"
            elif (i==(col - 1)):
                sep += "---:"
            else:
                sep += ":---:"
    table_str += f"{sep}\n"

    for r in range(0, (row)):
        row_val = ''
        for i in range(0, (col + 1)):
            row_val+="|"
            if not (i == col):
                if data:
                    row_val += str(data[r][i])
                else:
                    row_val += f"value_{i+1}"
        table_str += f"{row_val}\n"

    print(table_str)

    return table_str

if __name__ == '__main__':
    main()
