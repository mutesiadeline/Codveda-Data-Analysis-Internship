import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
df = pd.read_csv("churn-bigml-80.csv")

# 1. Display dataset information
print("Dataset Information:")
print(df.info())

# 2. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Check duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# 4. Remove duplicate rows
df = df.drop_duplicates()

print("\nDuplicates after removal:")
print(df.duplicated().sum())

# 5. Check categorical values
print("\nCategorical columns:")
print(df.select_dtypes(include="object").columns)

# Display cleaned dataset
print("\nCleaned Dataset Preview:")
print(df.head())

# Display first 5 rows
print(df.head())
# -------------------------------
# Summary Statistics
# -------------------------------

# Mean
print("Mean:")
print(df.mean(numeric_only=True))

# Median
print("\nMedian:")
print(df.median(numeric_only=True))

# Mode
print("\nMode:")
print(df.mode().iloc[0])

# Standard deviation
print("\nStandard Deviation:")
print(df.std(numeric_only=True))


# -------------------------------
# Data Distribution Visualization
# -------------------------------

# Histogram for numerical features
df.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()


# Boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.select_dtypes(include=['int64', 'float64']))
plt.xticks(rotation=90)
plt.title("Boxplot of Numerical Features")
plt.show()


# -------------------------------
# Correlation Analysis
# -------------------------------

# Select numerical columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Correlation matrix
correlation = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation)

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# 1. Bar Plot
# -------------------------------

# Count customers by churn status
plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

plt.legend(["Customers"])

plt.savefig("barplot_churn_distribution.png", bbox_inches="tight")

plt.show()



# -------------------------------
# 2. Line Chart
# -------------------------------

# Average total day minutes by account length groups
line_data = df.groupby("Account length")["Total day minutes"].mean()


plt.figure(figsize=(10,5))

plt.plot(
    line_data.index,
    line_data.values,
    marker="o",
    label="Average Day Minutes"
)


plt.title("Average Day Minutes by Account Length")
plt.xlabel("Account Length")
plt.ylabel("Average Total Day Minutes")

plt.legend()

plt.grid(True)

plt.savefig("linechart_day_minutes.png", bbox_inches="tight")

plt.show()



# -------------------------------
# 3. Scatter Plot
# -------------------------------

plt.figure(figsize=(8,5))


sns.scatterplot(
    data=df,
    x="Total day minutes",
    y="Total day charge",
    hue="Churn"
)


plt.title("Relationship Between Day Minutes and Day Charge")
plt.xlabel("Total Day Minutes")
plt.ylabel("Total Day Charge")

plt.legend(title="Churn Status")


plt.savefig("scatterplot_minutes_charge.png", bbox_inches="tight")

plt.show()
