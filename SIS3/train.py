import sys
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import seaborn as sns
from mlflow.tracking import MlflowClient
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, cross_val_score, train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
TARGET = "species"

EXPERIMENT_NAME = "penguin-species-classification"
MODEL_NAME = "penguin-classifier"

MODEL_TYPES = {"RandomForest": RandomForestClassifier, "ExtraTrees": ExtraTreesClassifier}

PARAM_GRID = {
    "model_type": list(MODEL_TYPES),
    "n_estimators": [10, 50, 100, 200],
    "max_depth": [2, 3, 5, 8, None],
    "min_samples_split": [2, 5, 10],
}

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

df = sns.load_dataset("penguins").dropna()
df["sex"] = df["sex"].map({"Female": 0, "Male": 1}) 
X, y = df[FEATURES], df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

grid = list(ParameterGrid(PARAM_GRID))
print(f"Number of models: {len(grid)}")

best = {"cv_f1": -1.0}

for i, params in enumerate(grid, start=1):
    params = dict(params)
    model_type = params.pop("model_type")

    with mlflow.start_run(run_name=f"{model_type}-{i:03d}"):
        model = MODEL_TYPES[model_type](**params, random_state=42)

        cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro").mean()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "cv_f1_macro": cv_f1,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
        }
        mlflow.log_params({"model_type": model_type, **params, "random_state": 42, "test_size": 0.2})
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(model, path)
            mlflow.log_artifact(str(path))

        print(f"[{i}/{len(grid)}] {model_type} {params} -> cv_f1={cv_f1:.3f}, test_f1={metrics['f1_macro']:.3f}")

        if cv_f1 > best["cv_f1"]:
            best = {"cv_f1": cv_f1, "model_uri": model_info.model_uri, "model": model,
                    "name": f"{model_type} {params}"}

print(f"\nThe best model: {best['name']} (cv_f1={best['cv_f1']:.3f})")

version = mlflow.register_model(best["model_uri"], MODEL_NAME)
MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version.version)
print(f"Registered model '{MODEL_NAME}', version {version.version} (alias: champion)")

joblib.dump(best["model"], "model.joblib")
print("The best model in model.joblib")