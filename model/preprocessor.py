import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pickle

def preprocess_data(data_path):
    # Load data
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop(columns=["math score"])
    y = df["math score"]

    # Column transformer for preprocessing
    numeric_features = ["reading score", "writing score"]
    categorical_features = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit and transform the training data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Save the transformed X_test and y_test as CSV files
    pd.DataFrame(X_test).to_csv("data/X_test_transformed.csv", index=False)
    pd.DataFrame(y_test, columns=["math score"]).to_csv("data/y_test.csv", index=False)

    # Save the preprocessor for later use in the app
    with open("model/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print("Data preprocessed and transformed X_test, y_test saved successfully.")
    return X_train, X_test, y_train, y_test
