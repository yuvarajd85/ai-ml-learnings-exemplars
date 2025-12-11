from polars import DataFrame
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    df: DataFrame = pl.read_csv("../resources/datasets/spam_dataset.csv")
    print(df.head(10))
    df_pd = df.to_pandas()

    vectorizer, model = train_model(df_pd)

    check_inbox("Are we still on for Meeting",vectorizer, model)
    check_inbox("Click to win money",vectorizer, model)
    check_inbox("Final Notice: Your account will be deleted – take action",vectorizer, model)


def train_model(df_pd):
    x, y = df_pd['subject'], df_pd['spam_label']

    vectorizer = CountVectorizer()
    x_vectorized = vectorizer.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_vectorized, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    print(f"Model Accuracy: {model.score(x_test, y_test):.2%}")

    return vectorizer, model

def check_inbox(message, vectorizer,model):
    # vectorizer = CountVectorizer()
    msg_vectorized =  vectorizer.transform([message])

    prediction = model.predict(msg_vectorized)[0]
    status = "!SPAM!" if prediction == 1 else "!!NORMAL!!"
    print(f"'{message}' -> {status}")

if __name__=='__main__':
    main()