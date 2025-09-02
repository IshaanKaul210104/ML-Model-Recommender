import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import sqlite3
from sklearn.datasets import (
    fetch_california_housing, load_diabetes,
    load_iris, load_wine, make_classification, make_friedman1, make_blobs
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, silhouette_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

# --- Setup database (runs once) ---
def init_db():
    conn = sqlite3.connect("feedback_logs.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT,
            user_model TEXT,
            recommended_model TEXT,
            avg_skewness REAL,
            avg_correlation_with_target REAL,
            multicollinearity_score REAL
        )
    """)
    conn.commit()
    conn.close()

# Call this once at start
init_db()

# --- Save a log entry to DB ---
def save_log(log_entry):
    conn = sqlite3.connect("feedback_logs.db")
    pd.DataFrame([log_entry]).to_sql("logs", conn, if_exists="append", index=False)
    conn.close()

# --- Load logs ---
def load_logs():
    conn = sqlite3.connect("feedback_logs.db")
    df = pd.read_sql("SELECT * FROM logs", conn)
    conn.close()
    return df

st.title("🔍 ML Model Recommender")

# ---------------------- Meta-Feature Functions -----------------------
def calculate_meta_features(X, y=None, task='regression'):
    meta = {}
    meta['avg_skewness'] = np.mean(np.abs(skew(X)))
    if task in ['regression', 'classification'] and y is not None:
        correlations = [np.corrcoef(X[col], y)[0, 1] if np.std(y) != 0 else 0 for col in X.columns]
        meta['avg_correlation_with_target'] = np.nanmean(np.abs(correlations))
    else:
        meta['avg_correlation_with_target'] = 0
    try:
        vif_matrix = np.linalg.pinv(np.corrcoef(X.T))
        meta['multicollinearity_score'] = np.trace(vif_matrix)
    except:
        meta['multicollinearity_score'] = 0
    return meta

# ---------------------- Recommendation Logic -----------------------
def recommend_model(meta, task='regression'):
    if task == 'regression':
        if meta['avg_skewness'] > 2 or meta['multicollinearity_score'] > 50:
            return RandomForestRegressor(), "Recommended Random Forest due to high skew/multicollinearity."
        elif meta['avg_correlation_with_target'] > 0.5:
            return Ridge(), "Recommended Ridge due to strong linear correlation with target."
        else:
            return LinearRegression(), "Recommended Linear Regression as skewness ({:.2f}) and correlation ({:.2f}) are moderate, indicating a linear model may suffice.".format(meta['avg_skewness'], meta['avg_correlation_with_target'])
    elif task == 'classification':
        if meta['multicollinearity_score'] > 50:
            return RandomForestClassifier(), "Recommended Random Forest due to high multicollinearity."
        elif meta['avg_correlation_with_target'] > 0.4:
            return LogisticRegression(), "Recommended Logistic Regression due to correlation with target."
        else:
            return RandomForestClassifier(), "Recommended Random Forest as default fallback."
    elif task == 'clustering':
        if meta['avg_skewness'] > 2:
            return DBSCAN(), "Recommended DBSCAN due to high skewness."
        else:
            return KMeans(n_clusters=3, random_state=42), "Recommended KMeans due to well-clustered structure."

# ---------------------- Training & Evaluation -----------------------
def train_and_evaluate(model, X, y=None, task='regression'):
    if task == 'regression':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        st.write("**R² Score:**", r2)
        st.write("**Mean Squared Error:**", mse)
        return {"model": model.__class__.__name__, "R2": r2, "MSE": mse}

    elif task == 'classification':
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            st.warning("⚠️ Stratified split failed — falling back to random split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write("**Accuracy:**", acc)
        return {"model": model.__class__.__name__, "Accuracy": acc}

    elif task == 'clustering':
        model.fit(X)
        labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
        score = silhouette_score(X, labels)
        st.write("**Silhouette Score:**", score)
        return {"model": model.__class__.__name__, "Silhouette Score": score}

# ---------------------- UI and Data Loading -----------------------
task = st.sidebar.selectbox("Select task type", ["regression", "classification", "clustering"])
data_option = st.sidebar.radio("Choose dataset type:", ("Use built-in", "Upload your own"))

data = None

# --- Upload Option ---
if data_option == "Upload your own":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("📂 Preview of uploaded dataset:")
        st.dataframe(data.head())

# --- Built-in Datasets ---
else:
    if task == 'regression':
        reg_dataset = st.selectbox("Choose a regression dataset:", ["California Housing", "Diabetes", "Friedman1"])
        if reg_dataset == "California Housing":
            data = fetch_california_housing(as_frame=True).frame
        elif reg_dataset == "Diabetes":
            data = load_diabetes(as_frame=True).frame
        elif reg_dataset == "Friedman1":
            X, y = make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=42)
            data = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
            data['target'] = y

    elif task == 'classification':
        cls_dataset = st.selectbox("Choose a classification dataset:", ["Iris", "Wine", "Make Classification"])
        if cls_dataset == "Iris":
            data = load_iris(as_frame=True).frame
        elif cls_dataset == "Wine":
            data = load_wine(as_frame=True).frame
        elif cls_dataset == "Make Classification":
            try:
                X, y = make_classification(
                    n_samples=1000, n_features=10, n_informative=6,
                    n_redundant=2, n_classes=3, n_clusters_per_class=1, random_state=42
                )
                data = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
                data['target'] = y
            except ValueError as e:
                st.error(f"Dataset generation failed: {e}")

    elif task == 'clustering':
        X, _ = make_blobs(n_samples=500, n_features=5, centers=3, random_state=42)
        data = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])

# --- If dataset is ready ---
if data is not None:
    st.write("📑 **Columns detected:**", list(data.columns))

    # --- Target column selection ---
    if task in ['regression', 'classification']:
        target_col = st.selectbox("🎯 Select target column", options=data.columns)
        y = data[target_col]
        X = data.drop(columns=[target_col])
    else:
        X = data
        y = None

    # --- Preprocessing ---
    X = X.select_dtypes(include=[np.number]).dropna()
    if y is not None:
        y = y.loc[X.index]

    st.subheader("🔧 Preprocessing Suggestions:")
    st.write("✅ No missing values detected." if not X.isnull().values.any() else "⚠️ Missing values found.")
    st.write("✅ All selected columns are numeric.")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # --- Meta Features ---
    meta = calculate_meta_features(X_scaled, y, task)
    st.subheader("📊 Meta-Features Summary:")
    st.json(meta)   # cleaner JSON-style output

    # --- Recommendation ---
    model, reason = recommend_model(meta, task)
    st.subheader(f"✅ Recommended model: `{model.__class__.__name__}`")
    st.info(f"📘 Reason: {reason}")

    # --- Train/Eval recommended ---
    st.subheader("📈 Recommended Model Performance")
    rec_result = train_and_evaluate(model, X_scaled, y, task)

    # --- Override Option ---
    override = st.radio("❓ Would you like to try a different model instead?", ("no", "yes"))

    if override == "yes":
        st.subheader("🔁 Choose an alternative model:")
        if task == 'regression':
            alt_models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(),
                'RandomForest': RandomForestRegressor()
            }
        elif task == 'classification':
            alt_models = {
                'LogisticRegression': LogisticRegression(),
                'RandomForest': RandomForestClassifier()
            }
        elif task == 'clustering':
            alt_models = {
                'KMeans': KMeans(n_clusters=3, random_state=42),
                'DBSCAN': DBSCAN()
            }

        selected_model_name = st.selectbox("Choose a model", list(alt_models.keys()))
        chosen_model = alt_models[selected_model_name]

        st.write(f"🔁 **Evaluating user-chosen model:** `{selected_model_name}`")
        user_result = train_and_evaluate(chosen_model, X_scaled, y, task)

        # --- Comparison ---
        st.subheader("📊 Model Comparison (Recommended vs User-Selected)")
        if task == "regression":
            df_comp = pd.DataFrame({
                "Model": [rec_result["model"], user_result["model"]],
                "R² Score": [rec_result["R2"], user_result["R2"]],
                "MSE (Lower is better)": [rec_result["MSE"], user_result["MSE"]]
            })
        elif task == "classification":
            df_comp = pd.DataFrame({
                "Model": [rec_result["model"], user_result["model"]],
                "Accuracy": [rec_result["Accuracy"], user_result["Accuracy"]]
            })
        elif task == "clustering":
            df_comp = pd.DataFrame({
                "Model": [rec_result["model"], user_result["model"]],
                "Silhouette Score": [rec_result["Silhouette Score"], user_result["Silhouette Score"]]
            })

        st.dataframe(df_comp.set_index("Model"))
        df_comp_melted = df_comp.melt("Model", var_name="Metric", value_name="Score")

        chart = alt.Chart(df_comp_melted).mark_bar().encode(
            x=alt.X('Model:N', title="Models"),
            y=alt.Y('Score:Q', title="Score"),
            color='Metric:N',
            column='Metric:N'
        ).properties(width=50, height=300)

        st.altair_chart(chart, use_container_width=True)

        # --- Logging ---
        log_entry = {
            "task": task,
            "user_model": selected_model_name,
            "recommended_model": model.__class__.__name__,
            **meta
        }
        save_log(log_entry)

    else:
        log_entry = {
            "task": task,
            "user_model": model.__class__.__name__,
            "recommended_model": model.__class__.__name__,
            **meta
        }
        save_log(log_entry)

    # --- Show All Logs (persistent across sessions) ---
    df_log = load_logs()
    if not df_log.empty:
        st.subheader("📝 Feedback Log (All Sessions, from DB)")
        st.dataframe(df_log)

        csv = df_log.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Feedback Log as CSV",
            data=csv,
            file_name='feedback_log.csv',
            mime='text/csv'
        )