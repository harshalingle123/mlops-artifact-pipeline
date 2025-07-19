import pickle
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score

def run_inference():
    # Load dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    # Load scaler and model
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_train.pkl', 'rb') as f:
        model = pickle.load(f)

    # Scale data
    X = scaler.transform(X)

    # Generate predictions
    y_pred = model.predict(X)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    print(f"Inference Accuracy: {accuracy:.4f}")
    print(f"Inference F1-Score: {f1:.4f}")

if __name__ == "__main__":
    run_inference()