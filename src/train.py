import json
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_config(config_path="config/config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def train_model():
    # Load dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load hyperparameters
    config = load_config()
    C = config['C']
    solver = config['solver']
    max_iter = config['max_iter']

    # Train model
    model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, multi_class='multinomial')
    model.fit(X_train, y_train)

    # Evaluate model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save model and scaler
    with open('model_train.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return model

if __name__ == "__main__":
    train_model()