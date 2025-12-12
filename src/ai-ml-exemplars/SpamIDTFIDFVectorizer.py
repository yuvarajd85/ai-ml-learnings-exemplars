'''
Created on 12/11/25 at 11:30 PM
By yuvarajdurairaj
Module Name SpamIDTFIDFVectorizer
'''
import os

import joblib
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch


def main():
    df = pl.read_csv(f"../resources/datasets/spam_dataset.csv")
    print(df.head())
    df_pd = df.to_pandas()
    bundle = train_model(df_pd)
    print_metrics(bundle)

    #Test Some message
    for msg in [
        "Are we still on for Meeting",
        "Click to win money",
        "Final Notice: Your account will be deleted – take action"
    ]:
        validate_message(msg, bundle)

    #Save Model Artifacts
    save_artifacts(bundle,out_dir="./artifacts")


def train_model(df_pd):
    x, y = df_pd["subject"], df_pd["spam_label"]
    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, test_size=0.2, stratify=y)

    #Pipeline: TF-IDF + Logistic Regression
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1,2), #Include bi-grams
            token_pattern=r"(?u)\b\w\w+\b", #Ignore 1-char tokens
            min_df=2 #Drop Ultra rare items
        )),
        ("lrclf",LogisticRegression(
            max_iter=1000,
            solver="liblinear", # Good Algorithm for small and sparse data
            class_weight="balanced"
        ))
    ])
    #Training the Model
    pipe.fit(x_train, y_train)

    #Predictions and evaluation data
    y_pred = pipe.predict(x_test)
    y_proba = pipe.predict_proba(x_test)[:, 1]

    return Bunch(
        model=pipe,
        X_test=x_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba
    )

def print_metrics(bundle):
    y_test,y_pred, y_proba = bundle.y_test, bundle.y_pred, bundle.y_proba

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )

    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.2%}")
    print(f"Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Confusion Matrix:")
    print(cm)

def validate_message(message, bundle, threshold: float = 0.50):
    proba = bundle.model.predict_proba([message])[:,1][0]
    label_predicted = 1 if proba >= threshold else 0
    status = "!SPAM!" if label_predicted == 1 else "!!!NORMAL!!!"
    print(f"'{message}' -> {status} (p_spam={proba:.3f}, threshold={threshold:.2f})")

def save_artifacts(bundle, out_dir="./artifacts"):
    os.makedirs(out_dir,exist_ok=True)
    # Save full pipeline - includes Tf-Idf + LR
    joblib.dump(bundle.model, os.path.join(out_dir, "spam_subject_pipeline.joblib"))
    print(f"Saved pipeline to {os.path.abspath(out_dir)}")


if __name__ == '__main__':
    main()
