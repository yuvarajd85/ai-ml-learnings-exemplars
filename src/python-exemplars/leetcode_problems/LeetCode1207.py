'''
Created on 10/26/2025 at 8:29 PM
By yuvaraj
Module Name: LeetCode1207
'''
from collections import Counter
from typing import Set

from dotenv import load_dotenv

load_dotenv()
'''
Given an array of integers arr, return true if the number of occurrences of each value in the array is unique or false otherwise.
Example 1:
Input: arr = [1,2,2,1,1,3]
Output: true
Explanation: The value 1 has 3 occurrences, 2 has 2 and 3 has 1. No two values have the same number of occurrences.
Example 2:
Input: arr = [1,2]
Output: false
Example 3:
Input: arr = [-3,0,1,-3,1,1,1,-3,10,0]
Output: true
'''

#Conventional implementation to get the occurences of each item in list
def get_occurences(in_list:list) -> dict:
    out_dict = {}
    for item in in_list:
        if item not in out_dict:
            out_dict[item] = 1
        else:
            out_dict[item] += 1
    print(f"Constructed dictionary from list with counted Values: {out_dict}")
    return out_dict

#Using Standard Libraries from Python Collections
def get_occurences_count(in_list:list)->dict:
    out_dict = dict(Counter(in_list))
    print(f"Constructed dictionary from list with counted Values: {out_dict}")
    return out_dict

def identify_unique_occurence(in_dict:dict):
    check_set = set()
    for val in in_dict.values():
        if val in check_set:
            return False
        else:
            check_set.add(val)
    return True

def main():
    #Case-1:
    arr1 = [1,2,2,1,1,3]
    print(f"Whether unique occurence or not: {identify_unique_occurence(get_occurences_count(arr1))}")
    #Case-2:
    arr2 = [1,2]
    print(f"Whether unique occurence or not: {identify_unique_occurence(get_occurences(arr2))}")
    #Case-3:
    arr3 = [-3,0,1,-3,1,1,1,-3,10,0]
    print(f"Whether unique occurence or not: {identify_unique_occurence(get_occurences(arr3))}")


if __name__ == '__main__':
    main()
