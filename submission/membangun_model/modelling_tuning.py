import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

train_path = "insurance_preprocessing/insurance_train_preprocessed.csv"
test_path = "insurance_preprocessing/insurance_test_preprocessed.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

# mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# dagsHub
dagshub.init(repo_owner='alishaanggranidi', repo_name='my-first-repo', mlflow=True)

mlflow.set_experiment("Insurance_Cost_Prediction_TuningGB")

n_estimators_range = [100, 200, 300]     
learning_rate_range = [0.01, 0.05, 0.1]  
max_depth_range = [3, 5]                

best_score = -np.inf
best_params = {}

for n_est in n_estimators_range:
    for lr in learning_rate_range:
        for depth in max_depth_range:
            run_name = f"gb_n{n_est}_lr{lr}_d{depth}"
            
            with mlflow.start_run(run_name=run_name):
                model = GradientBoostingRegressor(
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=depth,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)

                # log param and metrics
                mlflow.log_params({
                    "n_estimators": n_est, 
                    "learning_rate": lr,
                    "max_depth": depth,
                    "model": "GradientBoosting"
                })
                
                mlflow.log_metric("MSE", mse)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("R2", r2)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("MAPE", mape)

                print(f"Run {run_name} -> R2: {r2:.4f}")

                # save best model
                if r2 > best_score:
                    best_score = r2
                    best_params = {
                        "n_estimators": n_est, 
                        "learning_rate": lr,
                        "max_depth": depth
                    }

                    os.makedirs("models", exist_ok=True)
                    model_path = "models/best_model.joblib"
                    joblib.dump(model, model_path)
                    
                    mlflow.log_artifact(model_path, artifact_path="model")
                    print(f"New best model R2: {best_score:.4f} and params: {best_params}")

print("Best R2:", best_score)
print("Best Params:", best_params)