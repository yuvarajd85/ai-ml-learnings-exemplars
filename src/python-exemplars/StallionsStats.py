'''
Created on 2/21/26 at 4:01 PM
By yuvarajdurairaj
Module Name StallionsStats
'''
import polars as pl
from polars import DataFrame
from dotenv import load_dotenv
load_dotenv()

pl.Config.set_tbl_cols(200)

def main():
    process_batting_stats()
    process_bowling_stats()

def schema_trim(df_cols):
    return {col: str(col).strip() for col in df_cols}

def convert_col_to_str_dtype(col_name:str,df:DataFrame):
    return df.with_columns([
        pl.col(col_name).cast(pl.datatypes.Utf8).alias(col_name)
    ])

def process_batting_stats():
    f40: DataFrame = pl.read_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025 GPCL 40 Batting Records.xlsx")
    f40 = f40.rename(schema_trim(f40.columns))
    f40 = get_bat_agg(get_batter_data(f40))
    print(f40.head())
    f40.write_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025/processed/f40-batting.xlsx")

    t30: DataFrame = pl.read_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025 PMCL T30 Batting Records.xlsx")
    t30 = t30.rename(schema_trim(t30.columns))
    t30 = get_bat_agg(get_batter_data(t30))
    print(t30.head())
    t30.write_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025/processed/t30-batting.xlsx")

    t20_pmcl: DataFrame = pl.read_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025 PMCL T20 Batting Records.xlsx")
    t20_pmcl = t20_pmcl.rename(schema_trim(t20_pmcl.columns))
    t20_pmcl = convert_col_to_str_dtype("Avg",t20_pmcl)
    t20_del: DataFrame = pl.read_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025 DELAWARE CUP T20 Batting Records.xlsx")
    t20_del = t20_del.drop("Group ")
    t20_del = t20_del.rename(schema_trim(t20_del.columns))
    t20_del = convert_col_to_str_dtype("Avg", t20_del)
    t20_tur: DataFrame = pl.read_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025 TurboCup T20 Batting Records.xlsx")
    t20_tur = t20_tur.drop("Group ")
    t20_tur = t20_tur.rename(schema_trim(t20_tur.columns))
    t20_tur = convert_col_to_str_dtype("Avg", t20_tur)
    t20: DataFrame = pl.concat([t20_pmcl, t20_del, t20_tur])
    t20 = t20.rename(schema_trim(t20.columns))
    t20 = get_bat_agg(get_batter_data(t20))
    print(t20.head())
    t20.write_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025/processed/t20-batting.xlsx")

    overall_batting = get_bat_agg(pl.concat([f40,t30, t20]))
    print(overall_batting.head())
    overall_batting.write_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025/processed/overall-batting.xlsx")

def get_batter_data(df: DataFrame) -> pl.DataFrame:
    df = (df.select(["Player", "Mat", "Inns", "NO", "Runs", "4's", "6's", "50's", "100's", "HS", "SR", "Avg"])
    .with_columns([
        pl.col("Avg").cast(pl.datatypes.Utf8).alias("Avg")
    ])
    .with_columns([
        pl.when(pl.col("Avg") == "--").then(pl.lit(0.00)).otherwise(pl.col("Avg")).alias("Avg")
    ])
    .with_columns([
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
    ]))
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

def process_bowling_stats():
    f40 : DataFrame = pl.read_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025 GPCL 40 Bowling Records.xlsx")
    f40 = f40.rename(schema_trim(f40.columns)).select(['Player', 'Team', 'Mat', 'Inns', 'Overs', 'Runs', 'Wkts', 'BBF', 'Mdns', 'dots', 'Econ', 'Avg', 'SR', 'Hat-trick', '4w', '5w', 'Wides', 'Nb'])
    f40 = get_bowl_agg(get_bowler_data(f40))
    print(f40.head())
    f40.write_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025/processed/f40-bowling.xlsx")
    t30 : DataFrame = pl.read_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025 PMCL T30 Bowling Records.xlsx")
    t30 = t30.rename(schema_trim(t30.columns)).select(['Player', 'Team', 'Mat', 'Inns', 'Overs', 'Runs', 'Wkts', 'BBF', 'Mdns', 'dots', 'Econ', 'Avg', 'SR', 'Hat-trick', '4w', '5w', 'Wides', 'Nb'])
    t30 = get_bowl_agg(get_bowler_data(t30))
    print(t30.head())
    t30.write_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025/processed/t30-bowling.xlsx")
    t20_pmcl : DataFrame = pl.read_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025 PMCL T20 Bowling Records.xlsx")
    t20_del : DataFrame = pl.read_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025 DELAWARE CUP T20 Bowling Records.xlsx")
    t20_del = t20_del.drop("Group ")
    t20_tur : DataFrame = pl.read_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025 TurboCup T20 Bowling Records.xlsx")
    t20_tur = t20_tur.drop("Group ")
    t20 : DataFrame = pl.concat([t20_pmcl, t20_del, t20_tur])
    t20 = t20.rename(schema_trim(t20.columns)).select(['Player', 'Team', 'Mat', 'Inns', 'Overs', 'Runs', 'Wkts', 'BBF', 'Mdns', 'dots', 'Econ', 'Avg', 'SR', 'Hat-trick', '4w', '5w', 'Wides', 'Nb'])
    t20 = get_bowl_agg(get_bowler_data(t20))
    print(t20.head())
    t20.write_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025/processed/t20-bowling.xlsx")

    overall_bowl = get_bowl_agg(pl.concat([f40, t30, t20]))
    print(overall_bowl.head(20))
    overall_bowl.write_excel("/Users/yuvarajdurairaj/Documents/yuvi-personal/Stallions-ClubWork/2025/processed/overall-bowling.xlsx")

def get_bowler_data(df:DataFrame) -> pl.DataFrame:
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