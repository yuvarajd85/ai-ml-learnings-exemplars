'''
Created on 3/28/26 at 10:41 PM
By yuvarajdurairaj
Module Name HFRandomForrestClassifier
'''
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def load_data():
    dataset = load_dataset("scikit-learn/iris")

    # Fix: extract each feature column separately, then stack into 2D array
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    X = np.column_stack([dataset['train'][f] for f in features])
    y = np.array(dataset['train']['Species'])

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    target_names = ['setosa', 'versicolor', 'virginica']

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Scores
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    print(f"\n{'='*50}")
    print(f" Model: {name}")
    print(f"{'='*50}")
    print(f"  Test Accuracy      : {accuracy:.4f}")
    print(f"  CV Mean Accuracy   : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy, cv_scores.mean()


def main():
    # Load and split data
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Dataset size : {len(X)} samples")
    print(f"Train size   : {len(X_train)} | Test size: {len(X_test)}")
    print(f"Features     : {X.shape[1]} | Classes: {len(np.unique(y))}")

    # Define models
    models = {
        "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "SVM"                : SVC(kernel='rbf', random_state=42),
    }

    # Evaluate all models and collect results
    results = {}
    for name, model in models.items():
        accuracy, cv_mean = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results[name] = {"test_accuracy": accuracy, "cv_mean": cv_mean}

    # Summary
    print(f"\n{'='*50}")
    print(" Summary")
    print(f"{'='*50}")
    print(f"  {'Model':<25} {'Test Acc':>10} {'CV Mean':>10}")
    print(f"  {'-'*45}")
    for name, scores in results.items():
        print(f"  {name:<25} {scores['test_accuracy']:>10.4f} {scores['cv_mean']:>10.4f}")

    best = max(results, key=lambda k: results[k]['test_accuracy'])
    print(f"\n  Best Model: {best} ({results[best]['test_accuracy']:.4f})")


if __name__ == '__main__':
    main()