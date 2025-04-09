'''
Created on 4/7/2025 at 6:12 PM
By yuvaraj
Module Name: RandomOptions
'''
from dotenv import load_dotenv
import random

load_dotenv()


def main():
    for _ in range(100):
        print(random.choice(['X','O']))


if __name__ == '__main__':
    main()
