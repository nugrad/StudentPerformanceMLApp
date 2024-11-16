import pickle
from sklearn.linear_model import LinearRegression
from preprocessor import preprocess_data

def train_model(data_path="data/StudentsPerformance.csv"):
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

   

    print("Model trained and saved to model/model.pkl")

if __name__ == "__main__":
    train_model()
