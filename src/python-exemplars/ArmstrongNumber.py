
from typing import List


def main():
    number = 153
    exponent = len(str(number))

    digits = identify_digits(number, [])
    print(digits)

    digits_sum = sum(int(digit) ** exponent for digit in digits)

    print(digits_sum)

    print(f"The given number {number} is armstrong: {digits_sum == number}.")

def identify_digits(num:int, digits: List):
    exponent = len(str(num))

    if digits is None:
        digits = []

    if exponent == 1:
        digits.append(num)
        return digits

    denominator = 10 ** (exponent - 1)
    quotient = num // denominator
    remainder = num % denominator
    digits.append(quotient)
    return identify_digits(remainder, digits)

if __name__ == '__main__':
    main()