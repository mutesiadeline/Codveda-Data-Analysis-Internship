import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# 1. Load dataset
df = pd.read_csv("churn-bigml-80.csv")

# View data
print(df.head())
print(df.info())


# 2. Handle missing values
df = df.dropna()


# 3. Separate features and target
# Change "target" to your actual target column name
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 4. Handle categorical variables
categorical_columns = X.select_dtypes(include=['object']).columns

encoder = LabelEncoder()

for column in categorical_columns:
    X[column] = encoder.fit_transform(X[column])


# Encode target if it is categorical
if y.dtype == 'object':
    y = encoder.fit_transform(y)


# 5. Feature scaling
scaler = StandardScaler()

X = scaler.fit_transform(X)


# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 7. Create models

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}


results = {}


# 8. Train and evaluate models

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction, average='weighted')
    recall = recall_score(y_test, prediction, average='weighted')
    f1 = f1_score(y_test, prediction, average='weighted')

    results[name] = accuracy

    print("\n", name)
    print("--------------------")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print(classification_report(y_test, prediction))


# 9. Hyperparameter tuning (Random Forest)

parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}


grid_search = GridSearchCV(
    RandomForestClassifier(),
    parameters,
    cv=5,
    scoring='accuracy'
)


grid_search.fit(X_train, y_train)


print("\nBest Parameters:")
print(grid_search.best_params_)

print("Best Accuracy:")
print(grid_search.best_score_)


# 10. Compare models visually

plt.figure(figsize=(8,5))

plt.bar(results.keys(), results.values())

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Classification Model Comparison")

plt.xticks(rotation=45)

plt.show()