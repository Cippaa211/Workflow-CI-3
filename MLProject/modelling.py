import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ====================== Setup MLflow =======================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("titanic-logreg-basic")

# ====================== Load Data ==========================
df = pd.read_csv('./dataset_preprocessing/dataset_preprocessed_train.csv')

# ====================== Label Encoding =====================
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====================== Training ============================
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# ====================== Evaluation ==========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ====================== MLflow Logging ======================
with mlflow.start_run(run_name="logreg-basic"):
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 500)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Simpan model saja (tanpa log_artifacts untuk hindari error mlflow-artifacts)
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"[INFO] Training selesai ✅ | Akurasi: {acc:.4f} | F1 Score: {f1:.4f}")
