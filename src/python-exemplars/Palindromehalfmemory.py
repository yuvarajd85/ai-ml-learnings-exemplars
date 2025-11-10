'''
Created on 2/10/2025 at 10:47 PM
By yuvaraj
Module Name: Palindromehalfmemory
'''
from dotenv import load_dotenv

load_dotenv()


def main():
    # word = "malayalam"
    word = "hannah"
    char_array = list(word)
    word_len = len(char_array)

    for i in range(word_len//2):
        print(f"Left: {char_array[i]} Right: {char_array[word_len - i - 1]}")
        if char_array[i] != char_array[word_len - i - 1]:
            print(f"Not a palindrome")



if __name__ == '__main__':
    main()
