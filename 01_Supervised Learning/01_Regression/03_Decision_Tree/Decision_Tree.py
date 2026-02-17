import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


class DecisionTreeRegressionModel:

    def __init__(self, random_state=0):
        # Initialize model
        self.random_state = random_state
        self.model = DecisionTreeRegressor(random_state=self.random_state)
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
    # Predict on training data
    # ------------------------------
    def predict_all(self):
        return self.model.predict(self.x)

    # ------------------------------
    # Compute residuals table
    # ------------------------------
    def residuals_table(self):
        y_pred = self.predict_all()
        residuals = y_pred - self.y
        table = pd.DataFrame({
            'predictions': y_pred,
            'residuals': residuals
        })
        return table

    # ------------------------------
    # Predict a single value
    # ------------------------------
    def predict_single(self, value):
        value = np.array(value).reshape(1, -1)
        return self.model.predict(value)

    # ------------------------------
    # Plot regression curve
    # ------------------------------
    def plot_results(self, save_path="decision_tree_plot.png"):

        # create high resolution grid
        x_grid = np.arange(self.x.min(), self.x.max(), 0.01)
        x_grid = x_grid.reshape((len(x_grid), 1))

        # scatter original points
        plt.scatter(self.x, self.y, color='red')

        # plot predictions
        plt.plot(x_grid, self.model.predict(x_grid), color='blue')

        plt.title('Decision Tree Regression')
        plt.xlabel('Position level')
        plt.ylabel('Salary')

        # save figure (since no GUI)
        plt.savefig(save_path)
        plt.close()
