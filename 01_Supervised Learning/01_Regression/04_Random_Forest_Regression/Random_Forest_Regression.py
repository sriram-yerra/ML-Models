import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


class RandomForestRegressionModel:

    def __init__(self, n_estimators=10, random_state=0):
        # Initialize Random Forest model
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.x = None
        self.y = None

    # ------------------------------
    # Load dataset
    # ------------------------------
    def load_data(self, path):
        dataset = pd.read_csv(path)
        self.x = dataset.iloc[:, 1:-1].values
        self.y = dataset.iloc[:, -1].values
        return self.x, self.y

    # ------------------------------
    # Train model
    # ------------------------------
    def train(self):
        self.model.fit(self.x, self.y)

    # ------------------------------
    # Predict on entire dataset
    # ------------------------------
    def predict_all(self):
        return self.model.predict(self.x)

    # ------------------------------
    # Residual table
    # ------------------------------
    def residuals_table(self):
        y_pred = self.predict_all()
        residuals = y_pred - self.y

        table = pd.DataFrame({
            "predictions": y_pred,
            "residuals": residuals
        })
        return table

    # ------------------------------
    # Predict single value
    # ------------------------------
    def predict_single(self, value):
        value = np.array(value).reshape(1, -1)
        return self.model.predict(value)

    # ------------------------------
    # Plot regression curve
    # ------------------------------
    def plot_results(self, save_path="random_forest_plot.png"):

        # FIXED: use scalar min/max
        x_grid = np.arange(self.x.min(), self.x.max(), 0.01)
        x_grid = x_grid.reshape((len(x_grid), 1))

        # plot actual data
        plt.scatter(self.x, self.y, color='red')

        # plot predictions
        plt.plot(x_grid, self.model.predict(x_grid), color='blue')

        plt.title('Random Forest Regression')
        plt.xlabel('Position level')
        plt.ylabel('Salary')

        # Save (since terminal is non-GUI)
        plt.savefig(save_path)
        plt.close()
