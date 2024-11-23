'''
Created on 2/22/2023 at 9:59 PM
@author Yuvaraj Durairaj
using PyCharm
'''
from sklearn.linear_model import LinearRegression
def main():
    c=4
    r=3


    in_list = ['yuvi',"dharani","amora","anitra","yuvi","dharani","yuvi"]

    #List comprehension
    out_list = [str(n).capitalize() for n in in_list]

    print(out_list)

    #removing duplicates using dictionary comprehension and list comprehension
    # dedup_dict = [k for k in {n:1 for n in in_list}.keys()]
    dedup_dict = {n:1 for n in in_list}.keys()

    print(dedup_dict)

    #printing name 1000 times without using loop
    name = "x"

    name = name.replace("x","xxxxxxxxxx")
    name = name.replace("x","xxxxxxxxxx")
    name = name.replace("x","xxxxxxxxxx")
    name = name.replace("x","yuvi ")

    print(name)

    match_str = "pot"

    out_str =get_match(match_str)
    print(out_str)

def get_match(match_str):
    match match_str:
        case "port" : return "portfolio"
        case "port_id" : return "portfolio"
        case _: return "nothing"

if __name__ == '__main__':
    main()
