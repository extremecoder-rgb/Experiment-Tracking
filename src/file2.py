import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import dagshub
import os

# Initialize DAGsHub
dagshub.init(repo_owner='extremecoder-rgb', repo_name='Experiment-Tracking', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/extremecoder-rgb/Experiment-Tracking.mlflow")

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Model parameters
max_depth = 8
n_estimators = 5

# Set MLflow experiment
mlflow.set_experiment('Mlflow-Experiment-Wine-Classification')

with mlflow.start_run():
    # Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and params
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Create and save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("Confusion-matrix.png")
    plt.close()

    # Log confusion matrix image
    mlflow.log_artifact("Confusion-matrix.png")

    # Save and log model manually using joblib
    model_path = "random_forest_model.pkl"
    joblib.dump(rf, model_path)
    mlflow.log_artifact(model_path)

    # Optional: log current script file
    if os.path.exists(__file__):
        mlflow.log_artifact(__file__)

    # Set tags
    mlflow.set_tags({"Author": "Subhranil", "Project": "Wine Classification"})

    print("Accuracy:", accuracy)
