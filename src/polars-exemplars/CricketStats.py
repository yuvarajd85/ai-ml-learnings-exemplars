'''
Created on 1/11/2025 at 9:56 PM
By yuvaraj
Module Name CricketStats
'''
import polars as pl
from dotenv import load_dotenv
from polars import DataFrame
load_dotenv()


def main():
    pl.Config.set_tbl_cols(200)
    df_t20_bowl : DataFrame = pl.read_excel("../resources/datasets/Stats-2024.xlsx",sheet_name="Overall-Bowl-T20")

    print(df_t20_bowl.head())

    df_t20_bowl_agg = (
        df_t20_bowl.group_by("Player").agg(
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

    print(df_t20_bowl_agg.head(30))

    df_t20_bat: DataFrame = pl.read_excel("../resources/datasets/Stats-2024.xlsx",sheet_name="Overall-Bat-T20")

    print(df_t20_bat.head())

    df_t20_bat_agg = (df_t20_bat.group_by("Player").agg(
        pl.col("Mat").sum().alias("Mat"),
        pl.col("Inns").sum().alias("Inns"),
        pl.col("NO").sum().alias("NO"),
        pl.col("Runs").sum().alias("Runs")
    )).with_columns([
        (pl.col("Runs")/(pl.col("Inns")-pl.col("NO"))).alias("Avg")
    ]).sort(by=["Runs","Avg"],descending=[True,True])

    print(df_t20_bat_agg.head(30))

    df_bat_t30 =  pl.read_excel("../resources/datasets/Stats-2024.xlsx",sheet_name="PMCL-Bat-T30")
    df_bowl_t30 =  pl.read_excel("../resources/datasets/Stats-2024.xlsx",sheet_name="PMCL-Bowl-T30")
    df_bat_f40 = pl.read_excel("../resources/datasets/Stats-2024.xlsx", sheet_name="GPCL-Bat-F40")
    df_bowl_f40 = pl.read_excel("../resources/datasets/Stats-2024.xlsx", sheet_name="GPCL-Bowl-F40")

    df_bat_overall = pl.concat(items=[df_t20_bat,df_bat_t30,df_bat_f40],how='vertical')
    df_bowl_overall = pl.concat(items=[df_t20_bowl,df_bowl_t30,df_bowl_f40],how='vertical')

    df_bat_overall_agg = (df_bat_overall.group_by("Player").agg(
        pl.col("Mat").sum().alias("Mat"),
        pl.col("Inns").sum().alias("Inns"),
        pl.col("NO").sum().alias("NO"),
        pl.col("Runs").sum().alias("Runs")
    )).with_columns([
        (pl.col("Runs")/(pl.col("Inns")-pl.col("NO"))).alias("Avg")
    ]).sort(by=["Runs","Avg"],descending=[True,True])

    print(df_bat_overall_agg.head(30))


    df_bowl_overall_agg = (
        df_bowl_overall.group_by("Player").agg(
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

    print(df_bowl_overall_agg.head(30))

    #Writing All Dataframes
    df_t20_bowl_agg.write_excel("../resources/datasets/Stats-2024-T20-Bowl-Agg", worksheet="T20-Bowl-Agg")
    df_t20_bat_agg.write_excel("../resources/datasets/Stats-2024-T20-Bat-Agg",worksheet="T20-Bat-Agg")
    df_bowl_overall_agg.write_excel("../resources/datasets/Stats-2024-Overall-Bowl-Agg", worksheet="Bowl-Agg")
    df_bat_overall_agg.write_excel("../resources/datasets/Stats-2024-Overall-Bat-Agg",worksheet="Bat-Agg")



if __name__ == '__main__':
    main()
