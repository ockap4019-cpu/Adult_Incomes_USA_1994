# ============================================================
# 0. SETUP — Load libraries and inspect working environment
# Purpose: Prepare the environment, load dependencies, and confirm
#          that the dataset is accessible and correctly loaded.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set(style="whitegrid")

print(">>> Current working directory:")
print(os.getcwd())

print("\n>>> Files Python can see in this folder:")
print(os.listdir())

print("\n>>> Trying to load adult11.csv...")
df = pd.read_csv("adult11.csv")
print("\n>>> SUCCESS! First rows:")
print(df.head())


# ============================================================
# 1. DATA CLEANING — Handle missing values and simplify variables
# Purpose: Ensure data quality by removing missing values and
#          preparing categorical variables for analysis.
# ============================================================

df = df.replace('?', np.nan)
df_clean = df.dropna()

print("\n>>> Actual columns in df_clean:")
print(df_clean.columns.tolist())

print("\n>>> Cleaned dataset shape:")
print(df_clean.shape)

print("\n>>> Dataset info:")
print(df_clean.info())

print("\n>>> Salary distribution:")
print(df_clean['salary'].value_counts())

# Simplify native-country by grouping rare categories
country_counts = df_clean['native-country'].value_counts()
rare_countries = country_counts[country_counts < 500].index
df_clean['native-country-simplified'] = df_clean['native-country'].replace(rare_countries, 'Other')


# ============================================================
# 2. DEFINE VARIABLE TYPES — Numeric and categorical columns
# Purpose: Organise variables for EDA, statistical tests, and modelling.
# ============================================================

numeric_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

categorical_cols = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'gender', 'native-country-simplified', 'salary'
]


# ============================================================
# 3. DESCRIPTIVE STATISTICS — Summary of numeric and categorical data
# Purpose: Provide an overview of distributions and category frequencies.
# ============================================================

print("\n>>> Numerical summary statistics:")
print(df_clean[numeric_cols].describe().T)

print("\n>>> Categorical summary (value counts and percentages):")
for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(df_clean[col].value_counts())
    print("\nPercentages:")
    print((df_clean[col].value_counts(normalize=True) * 100).round(2))


# ============================================================
# 4. NUMERICAL EDA — Histograms and boxplots
# Purpose: Visualise distributions, detect skewness and outliers.
# ============================================================

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df_clean[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df_clean[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()


# ============================================================
# 5. CATEGORICAL EDA — Frequency plots and crosstabs
# Purpose: Explore category distributions and their relationship with salary.
# ============================================================

categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns:", categorical_columns)

print("\n=== Cardinality of categorical variables ===")
for col in categorical_columns:
    print(f"{col}: {df[col].nunique()} categories")

print("\n=== Percentage distribution per category ===")
for col in categorical_columns:
    print(f"\nColumn: {col}")
    print(df[col].value_counts(normalize=True).round(3))

# Countplots
for col in categorical_columns:
    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, y=col, order=df[col].value_counts().index)
    plt.title(f"Countplot of {col}")
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# Crosstabs vs salary
print("\n=== Relationship between categorical variables and salary ===")
for col in categorical_columns:
    if col != 'salary':
        print(f"\nCrosstab: {col} vs salary")
        print(pd.crosstab(df[col], df['salary'], normalize='index').round(3))


# ============================================================
# 6. TARGET RELATIONSHIPS — Visual comparisons with salary
# Purpose: Identify variables that differ between salary groups.
# ============================================================

sns.boxplot(x='salary', y='age', data=df_clean)
plt.title("Age vs Salary")
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='education', hue='salary', data=df_clean,
              order=df_clean['education'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Education vs Salary")
plt.show()

sns.boxplot(x='salary', y='hours-per-week', data=df_clean)
plt.title("Hours per Week vs Salary")
plt.show()


# ============================================================
# 7. CORRELATION HEATMAP — Numeric correlations
# Purpose: Detect linear relationships and potential multicollinearity.
# ============================================================

plt.figure(figsize=(10, 6))
sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# ============================================================
# 8. CHI-SQUARE TESTS — Categorical vs salary
# Purpose: Identify statistically significant associations.
# ============================================================

print("\n=== CHI-SQUARE TESTS FOR CATEGORICAL VARIABLES ===\n")

categorical_vars = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'gender', 'native-country'
]

for col in categorical_vars:
    contingency = pd.crosstab(df_clean[col], df_clean['salary'])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"Variable: {col}")
    print(f"Chi-square statistic = {chi2:.3f}, p-value = {p:.5f}\n")


# ============================================================
# 9. MANN-WHITNEY U TESTS — Numerical vs salary
# Purpose: Compare distributions between salary groups (non-parametric).
# ============================================================

print("\n=== MANN-WHITNEY U TESTS FOR NUMERICAL VARIABLES ===\n")

for col in numeric_cols:
    group1 = df_clean[df_clean['salary'] == '<=50K'][col]
    group2 = df_clean[df_clean['salary'] == '>50K'][col]
    stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
    print(f"Variable: {col}")
    print(f"Mann-Whitney U statistic = {stat:.3f}, p-value = {p:.5f}\n")


# ============================================================
# 10. LOGISTIC REGRESSION MODEL — Predict salary >50K
# Purpose: Build a predictive model and evaluate classification performance.
# ============================================================

print("\n=== LOGISTIC REGRESSION MODEL ===\n")

df_clean['salary_binary'] = df_clean['salary'].map({'<=50K': 0, '>50K': 1})

X = df_clean.drop(columns=['salary', 'salary_binary'])
y = df_clean['salary_binary']

X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42
)

logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ============================================================
# 11. ROC CURVE — Model performance
# Purpose: Evaluate discriminative ability of the model.
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 12. ODDS RATIOS — Feature importance
# Purpose: Interpret model coefficients in terms of effect size.
# ============================================================

coeffs = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Coefficient': logreg.coef_[0],
    'Odds Ratio': np.exp(logreg.coef_[0])
}).sort_values(by='Odds Ratio', ascending=False)

print("\nTop Predictive Features (Odds Ratios):\n")
print(coeffs.head(15))


# ============================================================
# 13. VIOLIN PLOTS — Distribution by salary group
# Purpose: Visualise distribution differences between salary categories.
# ============================================================

print("\n=== VIOLIN PLOTS FOR NUMERICAL VARIABLES BY SALARY ===\n")

for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.violinplot(
        data=df_clean,
        x='salary',
        y=col,
        hue='salary',
        palette='Set2',
        cut=0,
        legend=False
    )
    plt.title(f"Violin Plot of {col} by Salary Group")
    plt.xlabel("Salary")
    plt.ylabel(col)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


# ============================================================
# 14. VIF — Multicollinearity diagnostics
# ============================================================

# Copy encoded matrix
X_encoded_vif = X_encoded.copy()

# Convert everything to numeric
X_encoded_vif = X_encoded_vif.apply(pd.to_numeric, errors='coerce')

# Drop columns with ANY NaNs
X_encoded_vif = X_encoded_vif.dropna(axis=1)

# Drop columns with zero variance (constant)
X_encoded_vif = X_encoded_vif.loc[:, X_encoded_vif.std() > 0]

# Drop columns that are all zeros
X_encoded_vif = X_encoded_vif.loc[:, (X_encoded_vif != 0).any(axis=0)]

# Drop boolean columns
X_encoded_vif = X_encoded_vif.loc[:, X_encoded_vif.dtypes != bool]

# Drop columns with extremely low variance (almost constant)
X_encoded_vif = X_encoded_vif.loc[:, X_encoded_vif.std() > 1e-6]

# Compute VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X_encoded_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_encoded_vif.values, i)
    for i in range(X_encoded_vif.shape[1])
]

print("\n=== VARIANCE INFLATION FACTOR (VIF) ===\n")
print(vif_data.sort_values(by="VIF", ascending=False).head(15))

