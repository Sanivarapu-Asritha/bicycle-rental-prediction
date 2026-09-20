"""
Bicycle Rental Prediction Using Machine Learning
Reconstructed implementation based on the submitted project report.

Models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Dataset:
UCI Bike Sharing Dataset (2011-2012)
"""

import io
import zipfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def load_dataset():
    """Download and load the UCI Bike Sharing hourly dataset."""
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / "hour.csv"

    if not csv_path.exists():
        print("Downloading UCI Bike Sharing Dataset...")
        data = urllib.request.urlopen(DATA_URL).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(DATA_DIR)

    return pd.read_csv(csv_path)


def prepare_data(df):
    """Clean data and create time-related features."""
    df = df.copy()

    # Remove identifier columns that are not useful as direct predictors.
    df = df.drop(columns=["instant", "casual", "registered"], errors="ignore")

    # Convert the date column and derive useful temporal features.
    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"])
        df["year_month"] = df["dteday"].dt.to_period("M").astype(str)
        df["day_of_week"] = df["dteday"].dt.dayofweek
        df = df.drop(columns=["dteday"])

    # The target is total rental count.
    target = "cnt"
    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def build_preprocessor(X):
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_dataset()

    # Basic cleaning.
    df = df.drop_duplicates()
    df = df.dropna()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = build_preprocessor(X)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=12, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }

    results = []
    predictions = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5

        results.append(
            {
                "Model": name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
            }
        )
        predictions[name] = y_pred

        print(f"\n{name}")
        print(f"MAE : {mae:.2f}")
        print(f"MSE : {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # Plot actual vs predicted values for the Random Forest model.
    rf_pred = predictions["Random Forest"]

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, rf_pred, alpha=0.35)
    plt.xlabel("Actual Rental Count")
    plt.ylabel("Predicted Rental Count")
    plt.title("Random Forest: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    print("\nModel comparison saved to results/model_comparison.csv")
    print("Prediction plot saved to results/actual_vs_predicted.png")


if __name__ == "__main__":
    main()
