import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

MODEL_CONFIGS = [
    ('logistic', LogisticRegression, {'random_state': 42}, 0.75),
    ('forest', RandomForestClassifier, {'n_estimators': 100, 'random_state': 42}, 0.8),
    ('svm', LinearSVC, {'random_state': 42}, 0.75)
]

@pytest.fixture
def logistic_model():
    return LogisticRegression(random_state=42)

@pytest.fixture
def forest_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)

@pytest.fixture
def svm_model():
    return LinearSVC(random_state=42)

@pytest.fixture(scope="session")
def training_data():
    """Generate dataset once for all tests"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


class BaseModelTests:
    """Base class containing common tests for all models"""
    
    @pytest.fixture(autouse=True)
    def setup_model(self, request, training_data):
        """Setup model instance and training data"""
        self.X_train, self.X_test, self.y_train, self.y_test = training_data
        self.model = request.getfixturevalue(f"{self.model_name}_model")
    
    def test_model_initialization(self):
        """Test if model is properly initialized"""
        assert isinstance(self.model, self.expected_class)
        
    def test_input_shape(self):
        """Test if input data has correct shape"""
        assert self.X_train.shape[1] == 20
        assert len(self.y_train.shape) == 1
        
    def test_model_fitting(self):
        """Test if model can be fitted"""
        try:
            self.model.fit(self.X_train, self.y_train)
            fitted = True
        except Exception as e:
            fitted = False
        assert fitted, f"Model {self.model_name} failed to fit"
        
    def test_prediction_shape(self):
        """Test prediction shape after fitting"""
        self.model.fit(self.X_train, self.y_train)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(self.X_test)
            assert proba.shape == (len(self.X_test), 2)
        predictions = self.model.predict(self.X_test)
        assert len(predictions) == len(self.X_test)
        
    def test_model_performance(self):
        """Test if model meets minimum performance threshold"""
        self.model.fit(self.X_train, self.y_train)
        score = self.model.score(self.X_test, self.y_test)
        assert score > self.min_accuracy, \
            f"{self.model_name} accuracy {score:.3f} below threshold {self.min_accuracy}"

    def test_unfitted_model(self):
        """Test behavior of unfitted model"""
        with pytest.raises((NotFittedError, ValueError)):
            self.model.predict(self.X_test)

    @pytest.mark.parametrize("invalid_input", [
        np.array([[1, 2]]),  # Wrong features
        np.array([[np.nan] * 20]),  # NaN values
        np.array([[np.inf] * 20]),  # Infinite values
    ])
    def test_invalid_inputs(self, invalid_input):
        """Test model behavior with invalid inputs"""
        self.model.fit(self.X_train, self.y_train)
        with pytest.raises((ValueError, RuntimeError)):
            self.model.predict(invalid_input)

class TestLogisticRegression(BaseModelTests):
    model_name = "logistic"
    expected_class = LogisticRegression
    min_accuracy = 0.75
    
    def test_probability_calibration(self):
        """Test probability predictions (specific to logistic regression)"""
        self.model.fit(self.X_train, self.y_train)
        proba = self.model.predict_proba(self.X_test)
        assert np.all((proba >= 0) & (proba <= 1))
        assert np.allclose(np.sum(proba, axis=1), 1.0)

class TestRandomForest(BaseModelTests):
    model_name = "forest"
    expected_class = RandomForestClassifier
    min_accuracy = 0.8
    
    def test_feature_importance(self):
        """Test feature importance (specific to random forest)"""
        self.model.fit(self.X_train, self.y_train)
        importances = self.model.feature_importances_
        assert len(importances) == self.X_train.shape[1]
        assert np.all(importances >= 0)

class TestLinearSVC(BaseModelTests):
    model_name = "svm"
    expected_class = LinearSVC
    min_accuracy = 0.75
    
    def test_support_vectors(self):
        """Test model coefficients (specific to LinearSVC)"""
        self.model.fit(self.X_train, self.y_train)
        coef = self.model.coef_
        assert coef.shape[1] == self.X_train.shape[1]

class TestModelComparison:
    @pytest.mark.parametrize("name,model_class,params,expected_accuracy", MODEL_CONFIGS)
    def test_model_performance_comparison(self, training_data, name, model_class, params, expected_accuracy):
        """Compare performance across all models"""
        X_train, X_test, y_train, y_test = training_data
        model = model_class(**params)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        assert score > expected_accuracy, \
            f"{name} accuracy {score:.3f} below expected {expected_accuracy}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])