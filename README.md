# HOUSE-PRICE-PREDICTION
HOUSE PRICE PREDICTION USING CALIFORNIA DATASET
Here’s an enhanced and well-formatted **README.md** file with emojis, clear structure, and full explanations, tailored for your **California House Price Prediction** project:

---

# California House Price Prediction 🏠📊

This beginner-friendly machine learning project predicts house prices in California using the **California Housing Dataset**. It walks through every critical step in a typical data science pipeline — from data cleaning and outlier treatment to model training, tuning, and evaluation.

---

## Dataset Overview 📁

* **Total Rows:** `20,640`

* **Total Columns:** `10`

  * **Numerical Columns:** 9
  * **Categorical Column:** `ocean_proximity` (location type)

* **Target Variable:** `median_house_value` (California house prices)

---

## Steps Followed in the Project ✅

### 1. Data Cleaning 🧹

* **Missing Values:**

  * Only `total_bedrooms` had **207 null values**.
  * Imputed using **median** value (after checking for outliers).

* **Null Validation:** Confirmed no missing values remained after imputation.

---

### 2. Outlier Detection & Removal 🔍

* Visualized using **box plots**.
* Outliers were found in **6 out of 9 numerical columns**.
* Removed using **IQR method**, reducing the dataset from **20,640 ➝ 16,811 rows**.
* Rechecked: Outliers significantly reduced.

---

### 3. Feature Engineering 🛠️

Created 3 new meaningful features:

* `rooms_per_household = total_rooms / households`
* `bedrooms_per_room = total_bedrooms / total_rooms`
* `population_per_household = population / households`

These helped boost model performance by capturing more realistic relationships.

---

### 4. Encoding the Categorical Variable 🔤

* `ocean_proximity` was:

  * **One-hot encoded** for **Linear Regression**
  * **Label encoded** for tree-based models (Random Forest, XGBoost, etc.)

---

### 5. Model Building & Evaluation 🧠

Initial Models Trained:

| Model             | R² Score |
| ----------------- | -------- |
| Linear Regression | 0.51     |
| Random Forest     | 0.53     |
| XGBoost           | 0.52     |

After Feature Selection (top correlated features):

* `median_income`, `ocean_proximity`, `total_rooms`, `households`, `housing_median_age`

Still, R² scores remained around **0.52–0.53**.

---

### 6. Improved Feature Set + Re-training ⚡

* Included engineered features in training:

  * `rooms_per_household`, `bedrooms_per_room`, `population_per_household`
* Re-trained models and got improved results:

| Model         | R² Score |
| ------------- | -------- |
| Random Forest | 0.77     |
| XGBoost       | 0.80     |

However, both models showed signs of **overfitting**.

---

### 7. Tackling Overfitting 🔧

Tried models like:

* **LightGBM**
* **CatBoost**
* **XGBoost**
Despite high training performance, they also **overfit**.

#### Hyperparameter Tuning (Random Forest):

* Used manual tuning:

  * `max_depth = 10`
  * `min_samples_leaf = 5`
  * `max_features = 'sqrt'`
  * `n_estimators = 200`
  * `random_state = 42`

This configuration gave a **well-generalized model** with better bias-variance balance.

---

### 8. Final Evaluation 📈

* **Scatter Plot**: Actual vs. Predicted showed tight clustering near the diagonal.
* **Line Chart**: Actual vs. Predicted lines closely aligned.
* **Sample Predictions:**

  * 24% error
  * 55% accuracy
  * 2% error
  * 19% accuracy
  * 18% accuracy
  * Showing acceptable variance across cases.

---

## Final Model Summary 🏁

* **Final Model Used**: `Random Forest Regressor`
* **Final R² Score**: `~0.73`
* **Outlier-Free Cleaned Data**: `16,811 rows`
* **Encoding Used**: Label encoding (for final model)
* **Feature Engineered Columns**: 3
* **Best Hyperparameters**:

  * `max_depth=10`
  * `min_samples_leaf=5`
  * `n_estimators=200`
  * `max_features='sqrt'`
  * `random_state=42`

---

## Directory Structure 🗂️

```
.
├── data/
│   └── california_housing.csv
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
├── models/
│   └── random_forest_final.pkl
├── plots/
│   └── evaluation_visuals.png
├── README.md
```

---

## Tech Stack ⚙️

* **Language**: Python 3.x
* **Libraries**:

  * pandas
  * numpy
  * matplotlib, seaborn
  * scikit-learn
  * xgboost
  * catboost


