import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==================================
# STEP 1: LOAD DATASET
# ==================================

df = pd.read_csv("ecommerce.csv")

print("========== ORIGINAL DATASET ==========")
print(df.head())

# ==================================
# STEP 2: DATA CLEANING
# ==================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Fill missing numeric values with mean
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Fill missing categorical values with mode
categorical_cols = df.select_dtypes(include=["object", "string"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicate rows
df = df.drop_duplicates()

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())

# ==================================
# STEP 3: FEATURE SELECTION
# ==================================

# Features (Input Variables)
features = [
    "Quantity",
    "Unit_Price",
    "Delivery_Days",
    "Customer_Rating"
]

# Target (Output Variable)
target = "Total_Sales"

X = df[features]
y = df[target]

# ==================================
# STEP 4: TRAIN TEST SPLIT
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# ==================================
# STEP 5: MODEL TRAINING
# ==================================

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ==================================
# STEP 6: MODEL EVALUATION
# ==================================

results = {}

def evaluate_model(model_name, y_true, y_pred):

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{model_name} Performance")
    print("-" * 30)
    print("R2 Score :", round(r2, 3))
    print("RMSE     :", round(rmse, 3))

    results[model_name] = {
        "R2": r2,
        "RMSE": rmse
    }

# Evaluate all models
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Decision Tree", y_test, y_pred_dt)
evaluate_model("Random Forest", y_test, y_pred_rf)

# ==================================
# STEP 7: FIND BEST MODEL
# ==================================

best_model = max(results, key=lambda x: results[x]["R2"])

print("\n========== BEST MODEL ==========")
print("Model :", best_model)
print("R2    :", round(results[best_model]["R2"], 3))
print("RMSE  :", round(results[best_model]["RMSE"], 3))

# ==================================
# STEP 8: VISUALIZATION
# ==================================

predictions = {
    "Linear Regression": y_pred_lr,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf
}

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    predictions[best_model],
    alpha=0.6
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"{best_model}: Actual vs Predicted Sales")

plt.show()

# ==================================
# STEP 9: PROJECT INSIGHTS
# ==================================

print("\n========== PROJECT INSIGHTS ==========")

print(f"Best Performing Model: {best_model}")

if best_model == "Linear Regression":
    print("Linear Regression produced the highest R2 score.")
elif best_model == "Decision Tree":
    print("Decision Tree captured patterns better than other models.")
else:
    print("Random Forest achieved the best balance of accuracy and generalization.")

print("\nProject Completed Successfully!")