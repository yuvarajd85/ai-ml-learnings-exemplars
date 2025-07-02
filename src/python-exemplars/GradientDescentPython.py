'''
Created on 6/21/2025 at 2:24 AM
By yuvaraj
Module Name: GradientDescentPython
'''
from dotenv import load_dotenv

load_dotenv()


def main():
    # Data: y = 2x + 1
    X = [1, 2, 3, 4, 5]
    y = [3, 5, 7, 9, 11]

    # Initialize weights
    w = 0
    b = 0

    # Learning rate
    lr = 0.01

    # Perform 100 steps of gradient descent
    for epoch in range(100):
        dw = 0
        db = 0
        n = len(X)

        for i in range(n):
            y_pred = w * X[i] + b
            error = y_pred - y[i]
            dw += error * X[i]
            db += error

        dw /= n
        db /= n

        w -= lr * dw
        b -= lr * db

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: w = {w:.2f}, b = {b:.2f}")

    print(f"\nLearned model: y = {w:.2f}x + {b:.2f}")


if __name__ == '__main__':
    main()
