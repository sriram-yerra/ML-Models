# polynomial_regression_from_scratch.py

#--------------------------------------------------#
# Creating Class for Polynomial Regression         #
#--------------------------------------------------#
class MyPolynomialRegression:
    
    def __init__(self, degree=2):
        # Degree of the polynomial
        self.degree = degree
        
        # Model parameters
        self.coef_ = None
        self.intercept_ = None

    def _polynomial_features(self, X):
        # Generates polynomial features manually
        # Example (degree=3): x -> [x, x^2, x^3]
        X_poly = X.copy()
        for d in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly

    def fit(self, x_train, y_train):
        # Generate polynomial features
        x_poly = self._polynomial_features(x_train)
        
        # Add bias column (x0 = 1)
        x_poly = np.insert(x_poly, 0, 1, axis=1)

        # Normal Equation for Polynomial Regression
        # β = (XᵀX)⁻¹ Xᵀ y
        betas = np.linalg.inv(
            np.dot(x_poly.T, x_poly)
        ).dot(x_poly.T).dot(y_train)

        # Store intercept and coefficients
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, x_test):
        # Generate polynomial features for test data
        x_poly = self._polynomial_features(x_test)
        
        # Predict using learned coefficients
        y_pred = np.dot(x_poly, self.coef_) + self.intercept_
        return y_pred


#--------------------------------------------------#
# Importing the Libraries                           #
#--------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt


#--------------------------------------------------#
# Creating a synthetic dataset                     #
#--------------------------------------------------#
np.random.seed(10)

# Generate non-linear data
x = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 4 * x ** 3 + 2 * x ** 2 - 3 * x + 5 + np.random.randn(100, 1) * 10

# Visualizing original data
plt.scatter(x, y, s=15, label="Original Data")
plt.legend()
plt.show()


#--------------------------------------------------#
# Splitting the data into train and test sets       #
#--------------------------------------------------#
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=10
)


#--------------------------------------------------#
# Training custom Polynomial Regression model      #
#--------------------------------------------------#
model = MyPolynomialRegression(degree=3)
model.fit(x_train, y_train)


#--------------------------------------------------#
# Predicting results                               #
#--------------------------------------------------#
y_pred = model.predict(x_test)


#--------------------------------------------------#
# Evaluating model performance                     #
#--------------------------------------------------#
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
r2


#--------------------------------------------------#
# Visualizing model predictions                    #
#--------------------------------------------------#
x_sorted = np.sort(x, axis=0)
y_sorted_pred = model.predict(x_sorted)

plt.scatter(x, y, s=15, label="Original Data")
plt.plot(x_sorted, y_sorted_pred, color='red', label="Polynomial Fit")
plt.legend()
plt.show()


#--------------------------------------------------#
# Checking learned model parameters                #
#--------------------------------------------------#
model.coef_
model.intercept_
