import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from utils import generate_regression_data, split_data
from sklearn.metrics import r2_score, mean_squared_error

class GradientBoostingRegressionModel:
    def __init__(self, n_estimators=30, learning_rate=0.1, max_depth=3, random_state=0):
        # store params
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        # initialize model
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

        # placeholders
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    # ------------------------------
    # Load / generate data
    # ------------------------------
    def load_data(self, n_samples=100, test_ratio=0.25):
        X, y = generate_regression_data(n_samples)
        X_train, X_test, y_train, y_test = split_data(X, y, ratio=test_ratio)
        # flatten targets (sklearn expects 1D)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train[:, 0]
        self.y_test = y_test[:, 0]

    # ------------------------------
    # Train model
    # ------------------------------
    def train(self):
        self.model.fit(self.X_train, self.y_train)

    # ------------------------------
    # Predict
    # ------------------------------
    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    # ------------------------------
    # Plot results
    # ------------------------------
    def plot_results(self, save_path=None):
        # sort X for smooth curve
        indices = np.argsort(self.X_test[:, 0])
        xs = self.X_test[indices]
        ys = self.y_pred[indices]

        plt.figure(figsize=(8, 5))
        # actual data
        plt.plot(self.X_test, self.y_test, 'o', label='Actual')
        # predicted curve
        plt.plot(xs, ys, 'r', label='Gradient Boosting')

        plt.title("Gradient Boosting Regressor (sklearn)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid()
        plt.savefig(save_path)

    # ------------------------------
    # Evaluate model
    # ------------------------------
    def evaluate(self):
        r2 = r2_score(self.y_test, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        return {"R2": r2, "RMSE": rmse}
