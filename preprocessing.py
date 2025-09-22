# 1: Import dataset and explore basic info
import pandas as pd

# Load Titanic dataset (adjust path if needed)
df = pd.read_csv("titanic.csv")

print("First 5 rows:\n", df.head())
print("\nInfo:")
print(df.info())
print("\nSummary statistics:")
print(df.describe(include="all"))
print("\nMissing values before handling:\n", df.isnull().sum())

# ============================================================
# 2: Handle missing values
# ============================================================

# Age → fill with median
if 'Age' in df.columns:
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Age'] = df['Age'].fillna(df['Age'].median())

# Fare → fill with mean
if 'Fare' in df.columns:
    df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# Embarked → fill with mode
if 'Embarked' in df.columns and not df['Embarked'].mode().empty:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

print("\nMissing values after handling:\n", df.isnull().sum())

# ============================================================
# 3: Encode categorical features
# ============================================================

from sklearn.preprocessing import LabelEncoder

# Encode 'Sex'
if 'Sex' in df.columns:
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# Encode 'Embarked'
if 'Embarked' in df.columns:
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# One-hot encode 'Pclass'
if 'Pclass' in df.columns:
    df = pd.get_dummies(df, columns=['Pclass'], drop_first=True)

print("\nData after encoding:\n", df.head())

# ============================================================
# 4: Detect & Remove Outliers (before scaling)
# ============================================================

import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot before outlier removal
sns.boxplot(x=df['age'])
plt.title("Boxplot of Age (before outlier removal)")
plt.show()

sns.boxplot(x=df['fare'])
plt.title("Boxplot of Fare (before outlier removal)")
plt.show()

# --- Outlier removal using IQR ---

# For Age
Q1_age = df['age'].quantile(0.25)
Q3_age = df['age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
df = df[(df['age'] >= Q1_age - 1.5*IQR_age) & (df['age'] <= Q3_age + 1.5*IQR_age)]

# For Fare
Q1_fare = df['fare'].quantile(0.25)
Q3_fare = df['fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare
df = df[(df['fare'] >= Q1_fare - 1.5*IQR_fare) & (df['fare'] <= Q3_fare + 1.5*IQR_fare)]

print("\nShape after outlier removal:", df.shape)

# ============================================================
# 5: Scale numerical features
# ============================================================

from sklearn.preprocessing import StandardScaler

num_cols = ['age', 'fare']

# safety check: ensure numeric
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nScaled numerical columns:\n", df[num_cols].head())

# Boxplot after scaling
sns.boxplot(x=df['age'])
plt.title("Boxplot of Age (after cleaning & scaling)")
plt.show()

sns.boxplot(x=df['fare'])
plt.title("Boxplot of Fare (after cleaning & scaling)")
plt.show()