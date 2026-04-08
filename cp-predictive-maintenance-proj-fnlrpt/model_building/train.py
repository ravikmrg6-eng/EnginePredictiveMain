
# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_cap_proj_experiment")

api = HfApi()

Xtrain_path = "hf://datasets/ravikmrg6/CapStnProjMlopsPred/Xtrain.csv"
Xtest_path = "hf://datasets/ravikmrg6/CapStnProjMlopsPred/Xtest.csv"
ytrain_path = "hf://datasets/ravikmrg6/CapStnProjMlopsPred/ytrain.csv"
ytest_path = "hf://datasets/ravikmrg6/CapStnProjMlopsPred/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# Define base RandomForest type of classifier
rf_model = RandomForestClassifier(random_state=1)

# Hyperparameter grid
param_grid = {
    'randomforestclassifier__n_estimators': [110,251,501],
    'randomforestclassifier__max_depth':  [5,10,15]
    }

# Pipeline
model_pipeline = make_pipeline(preprocessor, rf_model)

with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=1, scoring='recall')
    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_recall", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Metrics - Corrected call to model_performance_classification_sklearn
    result = model_performance_classification_sklearn(model=best_model, predictors=Xtest, target=ytest)

    # Log metrics
    mlflow.log_metrics({
      "rf_accuracy": result["Accuracy"].iloc[0],
      "rf_model_recall": result["Recall"].iloc[0],
      "rf_model_precision": result["Precision"].iloc[0],
      "rf_model_f1": result["F1"].iloc[0]
    })

# Save the model locally
    model_path = "cs_pred_maintenance.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "ravikmrg6/CapStnProjMlopsPred"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="cs_pred_maintenance.joblib",
        path_in_repo="cs_pred_maintenance.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
