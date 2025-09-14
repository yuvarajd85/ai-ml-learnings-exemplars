'''
Created on 2/4/2025 at 12:09 AM
By yuvaraj
Module Name: bs4_stallions_club_stats
'''
from typing import Dict

import bs4
import polars as pl
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

output_filename = f"../resources/datasets/Stallions-Stats-2025.xlsx"

def main():
    pl.Config.set_tbl_cols(200)
    pl.Config.set_tbl_rows(200)
    batting_data : Dict = process_bowling_stats()
    bowling_data: Dict = process_batting_stats()
    total_data = batting_data | bowling_data

    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        for sheet_name, df in total_data.items():
            df.to_pandas().to_excel(writer, sheet_name=sheet_name, index=False)

def get_page_data(url:str) -> pl.DataFrame:
    response = requests.get(url)
    bs4_obj: bs4.BeautifulSoup = BeautifulSoup(response.text, features='html.parser')
    table = bs4_obj.find(name='table', attrs={'class': ['table table-striped table-active2 playersData sortable',
                                                        'table table-striped table-active4 playersData sortable',
                                                        'table table-striped table-active3 playersData sortable',
                                                        'table table-striped table-active3 playersData sortable dataTable no-footer'
                                                        ]})
    headers = []
    table_head = table.find('thead').find_all('th')
    for th in table_head:
        headers.append(th.text.strip())

    rows = []
    table_rows = table.find('tbody').find_all('tr')
    for tr in table_rows:
        row_values = tr.find_all('td')
        row = [cell.text.strip() for cell in row_values]
        rows.append(row)

    data = [dict(zip(headers, row)) for row in rows]
    df: pl.DataFrame = pl.DataFrame(data)
    return df

def process_batting_stats() -> Dict:
    f40_bat_urls = [f"https://cricclubs.com/GPCL/battingRecords.do?league=40&teamId=723&year=2024&clubId=48"]
    t30_bat_urls = [f"https://cricclubs.com/PMCL/battingRecords.do?league=20&teamId=496&year=2024&clubId=585"]
    t20_bat_urls = [f"https://cricclubs.com/GPCL/battingRecords.do?league=39&teamId=694&year=2024&clubId=48",
                     f"https://cricclubs.com/DelawareCup/battingRecords.do?league=13&teamId=233&year=2024&clubId=341",
                     f"https://cricclubs.com/TurboCup/battingRecords.do?league=6&teamId=73&year=2024&clubId=6930",
                     f"https://cricclubs.com/UCL2024/battingRecords.do?league=13&teamId=196&year=2024&clubId=1092362"
                     ]

    f40_bat_data = get_bat_agg(df=pl.concat(items=[get_batter_data(url) for url in f40_bat_urls], how='vertical'))
    print(f"{f40_bat_data.head()=}")
    t30_bat_data = get_bat_agg(df=pl.concat(items=[get_batter_data(url) for url in t30_bat_urls], how='vertical'))
    print(f"{t30_bat_data.head()=}")
    t20_bat_data = get_bat_agg(df=pl.concat(items=[get_batter_data(url) for url in t20_bat_urls], how='vertical'))
    print(f"{t20_bat_data.head()=}")
    bat_overall_data = get_bat_agg(df=pl.concat(items=[f40_bat_data,t30_bat_data,t20_bat_data]))
    print(f"{bat_overall_data.head(30)=}")

    bat_data_dict = {"F40-Bat": f40_bat_data, "T30-Bat": t30_bat_data, "T20-Bat" : t20_bat_data, "Bat-Overall" : bat_overall_data}
    return bat_data_dict


def get_batter_data(url: str) -> pl.DataFrame:
    df: pl.DataFrame = get_page_data(url)
    print(df.head())
    df = df.select(["Player", "Mat", "Inns", "NO", "Runs", "4's", "6's", "50's", "100's", "HS", "SR", "Avg"]).with_columns([
        pl.when(pl.col("Avg") == "--").then(pl.lit(0.00)).otherwise(pl.col("Avg")).alias("Avg")
    ]).with_columns([
        pl.col("Mat").cast(pl.datatypes.Int64).alias("Mat"),
        pl.col("Inns").cast(pl.datatypes.Int64).alias("Inns"),
        pl.col("NO").cast(pl.datatypes.Int64).alias("NO"),
        pl.col("Runs").cast(pl.datatypes.Int64).alias("Runs"),
        pl.col("4's").cast(pl.datatypes.Int64).alias("4's"),
        pl.col("6's").cast(pl.datatypes.Int64).alias("6's"),
        pl.col("50's").cast(pl.datatypes.Int64).alias("50's"),
        pl.col("100's").cast(pl.datatypes.Int64).alias("100's"),
        pl.col("HS").cast(pl.datatypes.Int64).alias("HS"),
        pl.col("SR").cast(pl.datatypes.Float64).alias("SR"),
        pl.col("Avg").cast(pl.datatypes.Float64).alias("Avg")
    ])
    return df

def get_bat_agg(df:pl.DataFrame) -> pl.DataFrame:
    df: pl.DataFrame = (
        df.group_by("Player").agg(
    pl.col("Mat").sum().alias("Mat"),
          pl.col("Inns").sum().alias("Inns"),
          pl.col("NO").sum().alias("NO"),
          pl.col("Runs").sum().alias("Runs")
        )
    ).with_columns([
        (pl.col("Runs") / (pl.col("Inns") - pl.col("NO"))).alias("Avg")
    ]).sort(by=["Runs","Avg"],descending=[True,True])
    return df

def process_bowling_stats() -> Dict:
    f40_bowl_urls = [f"https://cricclubs.com/GPCL/bowlingRecords.do?league=41&teamId=749&year=2025&clubId=48"]
    t30_bowl_urls = [f"https://cricclubs.com/PMCL/bowlingRecords.do?league=25&teamId=530&year=2025&clubId=585"]
    t20_bowl_urls = [f"https://cricclubs.com/PMCL/bowlingRecords.do?league=24&teamId=585&year=2025&clubId=585",
                     f"https://cricclubs.com/DelawareCup/bowlingRecords.do?league=14&teamId=260&year=2025&clubId=341",
                     f"https://cricclubs.com/TurboCup/bowlingRecords.do?league=7&teamId=83&year=2025&clubId=6930",
                     # f"https://cricclubs.com/UCL2024/bowlingRecords.do?league=13&teamId=196&year=2024&clubId=1092362"
                     ]

    f40_bowl_data = get_bowl_agg(df=pl.concat(items=[get_bowler_data(url) for url in f40_bowl_urls], how='vertical'))
    print(f"{f40_bowl_data.head()=}")

    t30_bowl_data = get_bowl_agg(df=pl.concat(items=[get_bowler_data(url) for url in t30_bowl_urls], how='vertical'))
    print(f"{t30_bowl_data.head()=}")

    t20_bowl_data = get_bowl_agg(df=pl.concat(items=[get_bowler_data(url) for url in t20_bowl_urls], how='vertical'))
    print(f"{t20_bowl_data.head()=}")

    bowl_overall_data = get_bowl_agg(df=pl.concat(items=[f40_bowl_data, t30_bowl_data, t20_bowl_data], how='vertical'))
    print(f"{bowl_overall_data.head(30)=}")

    bowl_data_dict = {"F40-Bowl": f40_bowl_data, "T30-Bowl": t30_bowl_data, "T20-Bowl": t20_bowl_data,
                     "Bowl-Overall": bowl_overall_data}
    return bowl_data_dict

def get_bowler_data(url:str) -> pl.DataFrame:
    df: pl.DataFrame = get_page_data(url)
    df = df.select(["Player","Mat","Inns","Overs","Runs","Wkts","BBF","Mdns","dots","Econ","Avg","SR","Hat-trick","4w","5w","Wides","Nb"]).with_columns([
        pl.col("Mat").cast(pl.datatypes.Int64).alias("Mat"),
        pl.col("Inns").cast(pl.datatypes.Int64).alias("Inns"),
        pl.col("Overs").cast(pl.datatypes.Float64).alias("Overs"),
        pl.col("Runs").cast(pl.datatypes.Int64).alias("Runs"),
        pl.col("Wkts").cast(pl.datatypes.Int64).alias("Wkts"),
        pl.col("Mdns").cast(pl.datatypes.Int64).alias("Mdns"),
        pl.col("dots").cast(pl.datatypes.Int64).alias("dots"),
        pl.col("Econ").cast(pl.datatypes.Float64).alias("Econ"),
        pl.col("Avg").cast(pl.datatypes.Float64).alias("Avg"),
        pl.col("SR").cast(pl.datatypes.Float64).alias("SR"),
        pl.col("Hat-trick").cast(pl.datatypes.Int64).alias("Hat-trick"),
        pl.col("4w").cast(pl.datatypes.Int64).alias("4w"),
        pl.col("5w").cast(pl.datatypes.Int64).alias("5w"),
        pl.col("Wides").cast(pl.datatypes.Int64).alias("Wides"),
        pl.col("Nb").cast(pl.datatypes.Int64).alias("Nb")
    ])
    return df

def get_bowl_agg(df:pl.DataFrame) -> pl.DataFrame:
    df_bowl_agg = (
        df.group_by("Player").agg(
            pl.col("Mat").sum().alias("Mat"),
            pl.col("Inns").sum().alias("Inns"),
            pl.col("Overs").sum().alias("Overs"),
            pl.col("Runs").sum().alias("Runs"),
            pl.col("Wkts").sum().alias("Wkts"),
            pl.col("BBF").min().alias("BBF"),
            pl.col("Mdns").sum().alias("Mdns"),
            pl.col("dots").sum().alias("dots")
        )
    ).with_columns([
        (pl.col("Runs") / pl.col("Overs")).alias("Econ"),
        (pl.col("Runs")/ pl.col("Wkts")).alias("Avg")
    ]).sort(by=["Wkts","Econ","Avg"],descending=[True,False,False])
    return df_bowl_agg

if __name__ == '__main__':
    main()
