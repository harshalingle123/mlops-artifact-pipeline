import pytest
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from src.train import load_config, train_model

def test_config_loading():
    config = load_config()
    assert isinstance(config, dict), "Config should be a dictionary"
    assert 'C' in config, "C parameter missing"
    assert 'solver' in config, "solver parameter missing"
    assert 'max_iter' in config, "max_iter parameter missing"
    assert isinstance(config['C'], float), "C should be float"
    assert isinstance(config['solver'], str), "solver should be string"
    assert isinstance(config['max_iter'], int), "max_iter should be integer"

def test_model_creation():
    model = train_model()
    assert isinstance(model, LogisticRegression), "Model should be LogisticRegression"
    assert hasattr(model, 'coef_'), "Model should be fitted (coef_ exists)"
    assert hasattr(model, 'classes_'), "Model should be fitted (classes_ exists)"
    assert os.path.exists('scaler.pkl'), "Scaler file should be created"

def test_model_accuracy():
    digits = load_digits()
    X, y = digits.data, digits.target
    model = train_model()
    accuracy = model.score(X, y)
    assert accuracy > 0.8, f"Model accuracy {accuracy} is below threshold 0.8"