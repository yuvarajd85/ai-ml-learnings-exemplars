'''
Created on 8/29/2022 at 11:08 PM
@author Yuvaraj Durairaj
using PyCharm
'''
import os
import bs4 as bs
import pandas as pd
import requests
import datetime as dt

def main():
    start_time = dt.datetime.now()
    url = f"https://cricclubs.com/GPCL/listMatches.do?league=33&year=2022&clubId=48"
    page = requests.get(url)
    soup = bs.BeautifulSoup(page.content,'html.parser')
    urls = [f"https://cricclubs.com{a['href']}" for a in soup.find_all("a",href=True) if "viewScorecard" in a["href"]]
    urls = list({s:"" for s in urls}.keys())
    results = []

    for url in urls:
        print(url)
        scorecard = requests.get(url)
        score_soup = bs.BeautifulSoup(scorecard.content,"html.parser")
        team_names = score_soup.find_all(attrs={"class":"teamName"})
        teams_played = " Vs ".join(list(map(lambda s : s.text.strip(), team_names)))
        match_details = score_soup.find(attrs={"class":"match-detail-table"}).find("table")
        mom = "".join([a.text.strip() for a in match_details.find_all("a") if "viewPlayer" in a['href']])
        table_details = match_details.find("tbody").find_all("tr")
        series = table_details[0].find_all("th")[1].text.strip()
        match_date = table_details[1].find_all("th")[1].text.strip()

        result = {
            "Date" : f"{match_date}",
            "Series" : f"{series}",
            "Match" : f"{teams_played}",
            "Player_Of_The_Match" : f"{mom}"
        }

        results.append(result)

    print(results)

    df = pd.DataFrame(results).drop_duplicates(subset=["Date","Series","Match","Player_Of_The_Match"]).reset_index(drop=True)
    df = df[df["Player_Of_The_Match"] > ""]

    print(df.head(20))

    df.to_excel(f"C://Users//{str(os.getlogin()).strip()}//Desktop//GPCLT10MOM.xlsx",index=False)

    print(f"Total Execution Time: {dt.datetime.now() - start_time}")

if __name__ == '__main__':
    main()