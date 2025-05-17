'''
Created on 5/14/2025 at 6:07 PM
By yuvaraj
Module Name: SchemaEncoder
'''
from dataclasses import dataclass
from typing import get_type_hints

import polars as pl
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Student:
    id: int
    name: str
    term_fees: float
    total_marks: float
    avg_marks: float
    position: int

data_type_dict = {
    str : pl.Utf8,
    int : pl.Int64,
    float : pl.Float64,
    bool: pl.Boolean
}

def main():
    stud1 = Student(1,"Student1",250.50,575.00,92.50,2)

    schema_dict = {field: data_type_dict.get(field_type) for field, field_type in get_type_hints(stud1).items()}

    print(schema_dict)


if __name__ == '__main__':
    main()
