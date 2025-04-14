# 🧠 Regression Model Recommender

This project automatically analyzes a regression dataset and **recommends the most suitable machine learning regression model** based on the dataset's statistical characteristics — no need to try every model manually!

## 🔍 Project Goal

Instead of training all possible regression models to find the best one, this project aims to intelligently recommend the most suitable model by analyzing:

- Skewness of features
- Correlation with target variable
- Multicollinearity of input features

Based on these, it chooses between models like:
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

## 📂 How It Works

1. Upload your dataset (CSV) or use built-in sklearn datasets.
2. The script performs statistical analysis of the features.
3. Based on a simple decision heuristic, it recommends a model.
4. The recommended model is trained and evaluated using R² and RMSE scores.

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost (optional, for additional accuracy)

## 🚀 How to Use

1. Clone the repository or open the `Regression_Model_Recommender.ipynb` in Google Colab.
2. Upload your dataset, or test with built-in datasets like:
   - `fetch_california_housing()`
   - `load_diabetes()`
   - `make_friedman1()`
3. Enter your target column name when prompted.
4. View the recommended model and performance metrics.

## 📊 Lets say you uploaded a dataset which should ideally have the best prediction results with RandomForest Regression, then your output will be like this : 

Avg skewness: 0.36 Avg correlation with target: 0.34 Multicollinearity score: 470.07

✅ Recommended model: RandomForestRegressor

📊 Model Evaluation: R² Score: 0.95 RMSE: 0.58

## 📁 Sample Datasets

Some examples tested:
- California Housing → Random Forest
- Diabetes → Ridge
- Friedman1 → Random Forest
- Simple Linear Dataset → Linear Regression

## ✅ Future Work

- Add a GUI or web-based interface.
- Train a meta-model using meta-features to predict the best algorithm.
- Support for classification models.

## 🙋‍♂️ Author

Made with ❤️ by Ishaan Kaul.  
If you like this project, feel free to ⭐ it and share.
