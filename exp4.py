
# 1.  IMPORTS & DATA GENERATION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)                             # reproducible results

data = {
    "ID":        range(1, 11),
    "Age":       np.random.randint(18, 65, size=10),
    "Income":    np.random.randint(30_000, 90_000, size=10),
    "Gender":    ["Male", "Female", "Male", "Female", "Male",
                  "Female", "Male", "Female", "Male", "Male"],
    "Education": ["High School", "Bachelor", "Master", "PhD", "Bachelor",
                  "Master", "Bachelor", "PhD", "High School", "Master"]
}

df = pd.DataFrame(data)

# 2.  QUICK EDA INSPECTION

print("First five rows:")
print(df.head(), "\n")

print("Descriptive statistics (numeric cols):")
print(df.describe(), "\n")

print("Missing-value counts:")
print(df.isnull().sum(), "\n")

print("Unique values — Gender:", df["Gender"].unique())
print("Unique values — Education:", df["Education"].unique(), "\n")


# 3.  COLUMN PICKING & ROW FILTERING

# 3-A. Selecting only Age & Income
selected_cols = df[["Age", "Income"]]
print("Selected numeric columns:\n", selected_cols.head(), "\n")

# 3-B. Adults older than 30
filtered_age = df[df["Age"] > 30]
print("People older than 30:\n", filtered_age.head(), "\n")

# 3-C. Male respondents with a Master’s degree
male_masters = df[(df["Gender"] == "Male") & (df["Education"] == "Master")]
print("Male respondents w/ Master’s:\n", male_masters.head(), "\n")

# 4.  BASIC VISUALISATIONS

plt.figure(figsize=(5, 3))
plt.hist(df["Age"], bins=5, edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(4, 3))
plt.boxplot(df["Income"])
plt.title("Income Distribution")
plt.ylabel("Annual Income (₹)")
plt.tight_layout()
plt.show()

gender_counts = df["Gender"].value_counts()
gender_counts.plot(kind="bar", color="skyblue", figsize=(4, 3))
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

education_counts = df["Education"].value_counts()
education_counts.plot(kind="pie",
                      autopct="%1.1f%%",
                      figsize=(4, 4),
                      legend=False,
                      labels=education_counts.index,
                      startangle=90)
plt.title("Education Distribution")
plt.ylabel("")           # hides the default ‘y’ label
plt.tight_layout()
plt.show()
