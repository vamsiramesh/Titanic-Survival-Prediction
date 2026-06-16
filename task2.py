import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
X = train.drop("Survived", axis=1)
y = train["Survived"]
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns
categorical_features = X.select_dtypes(
    include=["object"]
).columns
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    random_state=42
)
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)
pipeline.fit(X, y)
scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)
print("\nModel Accuracy:")
print(f"Mean Accuracy = {scores.mean():.4f}")
predictions = pipeline.predict(test)
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})
submission.to_csv(
    "titanic_submission.csv",
    index=False
)
print("\nPrediction file created successfully!")
print("Saved as: titanic_submission.csv")
print("\nFirst 10 Predictions:")
print(submission.head(10))