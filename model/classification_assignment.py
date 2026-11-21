import json
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


SEED = 42
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "model"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]


def download_dataset():
    raw_path = PROJECT_DIR / "adult.data"
    if not raw_path.exists():
        urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            raw_path,
        )
    data = pd.read_csv(raw_path, header=None, names=COLUMNS, na_values=" ?", skipinitialspace=True)
    data["income"] = data["income"].str.strip().str.replace(".", "", regex=False)
    data = data.drop_duplicates()
    data["target"] = data["income"].map({"<=50K": 0, ">50K": 1})
    data = data.drop(columns=["income"])
    return data


def make_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def evaluate_model(name, pipeline, X_test, y_test):
    predictions = pipeline.predict(X_test)
    probability = pipeline.predict_proba(X_test)[:, 1]
    return {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "AUC": round(roc_auc_score(y_test, probability), 4),
        "Precision": round(precision_score(y_test, predictions, zero_division=0), 4),
        "Recall": round(recall_score(y_test, predictions, zero_division=0), 4),
        "F1": round(f1_score(y_test, predictions, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, predictions), 4),
    }


def plot_results(results):
    sns.set_theme(style="whitegrid")
    metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    results_for_plot = pd.DataFrame(results).set_index("ML Model Name")
    ax = results_for_plot[metrics].plot(kind="bar", figsize=(12, 7), ylim=(0, 1))
    ax.set_title("Classification Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "model_comparison.png", dpi=180)
    plt.close()


def main():
    data = download_dataset()
    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    test_data = X_test.copy()
    test_data["target"] = y_test
    test_data.to_csv(PROJECT_DIR / "test_data.csv", index=False)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = make_preprocessor(numeric_features, categorical_features)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    results = []
    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)
        results.append(evaluate_model(name, pipeline, X_test, y_test))

    results_df = pd.DataFrame(results)
    results_df.to_csv(PROJECT_DIR / "model_results.csv", index=False)
    (PROJECT_DIR / "model_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    plot_results(results)

    print("Dataset shape:", data.shape)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("\nClass distribution (all data):")
    print(y.value_counts().rename({0: "<=50K", 1: ">50K"}).to_string())
    print("\nEvaluation results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
