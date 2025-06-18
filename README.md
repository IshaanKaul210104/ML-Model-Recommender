🤖 ML Model Recommender
This project automatically analyzes any dataset and recommends the most suitable machine learning model (regression, classification, or clustering) based on the dataset’s statistical characteristics — no need to try every model manually!

🎯 Project Goal
Instead of blindly trying every algorithm, this tool intelligently analyzes your dataset and recommends the most suitable ML model by examining:

🔁 Task Type (Regression, Classification, Clustering)

📈 Skewness of features

📊 Correlation with target (for supervised tasks)

🧮 Multicollinearity (via Variance Inflation Factor)

🧠 Heuristic rules for model suitability

🧠 Supported Models
📌 Regression:
Linear Regression

Ridge Regression

Decision Tree Regressor

Random Forest Regressor

XGBoost Regressor

📌 Classification:
Logistic Regression

Decision Tree Classifier

Random Forest Classifier

XGBoost Classifier

📌 Clustering:
KMeans

Agglomerative Clustering

DBSCAN

⚙️ How It Works
📂 Upload your dataset (CSV) or choose from built-in sklearn datasets.

🧪 The system performs:

Statistical analysis of features

Task type detection

Meta-feature extraction

🧠 A heuristic logic block selects the best-suited model.

📊 The recommended model is trained (if applicable) and evaluated.

🚀 How to Use
Clone this repository or open ML_Model_Recommender.ipynb in Google Colab.

Choose a built-in dataset or upload your own CSV.

Specify your task type and target column (if supervised).

Run the notebook to view:

✅ Recommended model

📈 Evaluation metrics (R², RMSE, Accuracy, Silhouette Score)

📁 Sample Outputs
🧪 Regression Example:
yaml
Copy
Edit
Avg skewness: 0.36  
Avg correlation with target: 0.34  
VIF score: 470.07

✅ Recommended Model: RandomForestRegressor  
📊 R² Score: 0.95  
📉 RMSE: 0.58
🧪 Classification Example:
yaml
Copy
Edit
✅ Recommended Model: RandomForestClassifier  
📊 Accuracy: 94%  
🎯 Precision: 92%  
🏷️ F1-score: 93%
🧪 Clustering Example:
yaml
Copy
Edit
✅ Recommended Model: KMeans  
📈 Silhouette Score: 0.67
📦 Tech Stack
Python

Pandas, NumPy

Scikit-learn

XGBoost

Matplotlib / Seaborn (optional for visualizations)

📊 Built-in Datasets (for testing)
fetch_california_housing()

load_diabetes()

load_iris()

load_breast_cancer()

make_classification()

make_regression()

make_blobs()

make_friedman1()

🔮 Future Work
Train a proper meta-model using dataset meta-features

Add support for time series model recommendations

Build a web-based GUI using Streamlit or Gradio

Benchmark against AutoML solutions

🙋‍♂️ Author
Made with ❤️ by Ishaan Kaul
If you like this project, feel free to ⭐ it and share!
