# lasso_regression_comparison.py

#--------------------------------------------------#
# Importing the Libraries                           #
#--------------------------------------------------#
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np


#--------------------------------------------------#
# Creating a synthetic regression dataset          #
#--------------------------------------------------#
x, y = make_regression(
    n_samples=100,
    n_features=1,
    n_informative=1,
    n_targets=1,
    noise=20,
    random_state=13
)

# Visualizing the generated dataset
plt.figure(figsize=[4.5, 3])
plt.scatter(x, y)
plt.show()


#--------------------------------------------------#
# Calculating intercept and slope values            #
# for Linear Regression model                      #
#--------------------------------------------------#
from sklearn.linear_model import LinearRegression

# Create Linear Regression object
regressor = LinearRegression()

# Fit model to data
regressor.fit(x, y)

# Print slope (coefficient) and intercept
print(regressor.coef_)
print(regressor.intercept_)


#--------------------------------------------------#
# Calculating intercept and slope values            #
# for Lasso Regression model (Alpha = 10)           #
#--------------------------------------------------#
from sklearn.linear_model import Lasso

# Lasso Regression with moderate regularization
lasso_reg = Lasso(alpha=10)

# Fit model to data
lasso_reg.fit(x, y)

# Print slope and intercept
print(lasso_reg.coef_)
print(lasso_reg.intercept_)


#--------------------------------------------------#
# Calculating intercept and slope values            #
# for Lasso Regression model (Alpha = 100)          #
#--------------------------------------------------#
lasso_reg2 = Lasso(alpha=100)

# Fit model with strong regularization
lasso_reg2.fit(x, y)

# Print slope and intercept
print(lasso_reg2.coef_)
print(lasso_reg2.intercept_)


#--------------------------------------------------#
# Comparing all models graphically                  #
#--------------------------------------------------#
plt.figure(figsize=[5.5, 4])

# Plot original data points
plt.plot(x, y, 'b.')

# Linear Regression prediction (alpha = 0)
plt.plot(x, regressor.predict(x), color='red', label='alpha=0')

# Lasso Regression predictions with different alphas
plt.plot(x, lasso_reg.predict(x), color='green', label='alpha=10')
plt.plot(x, lasso_reg2.predict(x), color='orange', label='alpha=100')

# Show legend and plot
plt.legend()
plt.show()


#--------------------------------------------------#
# Creating our own Lasso Regression class           #
#--------------------------------------------------#
class MyLassoRegression:
    
    def __init__(self, alpha=0.1):
        # Constructor for MyLassoRegression class.
        # Initializes regularization parameter (alpha)
        # and sets slope (m) and intercept (b) as None.
        self.alpha = alpha
        self.m = None
        self.b = None
        
    def fit(self, x_train, y_train):
        # Fits the Lasso Regression model using a
        # simplified closed-form–inspired computation
        # (NOTE: true Lasso is usually solved via optimization)

        num = 0  # Numerator for slope calculation
        den = 0  # Denominator for slope calculation
        
        # Loop through each training example
        for i in range(x_train.shape[0]):
            num = num + (
                (y_train[i] - y_train.mean()) *
                (x_train[i] - x_train.mean())
            )
            den = den + (
                (x_train[i] - x_train.mean()) *
                (x_train[i] - x_train.mean())
            )
        
        # Lasso-adjusted slope approximation
        # Uses sign-based penalty (soft thresholding idea)
        self.m = (num - self.alpha * np.sign(num)) / den

        # Intercept calculation
        self.b = y_train.mean() - (self.m * x_train.mean())
        
        # Print learned parameters
        print(self.m, self.b)
    
    def predict(self, x_test):
        # Predict target values using learned parameters
        return self.m * x_test + self.b


#--------------------------------------------------#
# Testing custom Lasso Regression implementation   #
#--------------------------------------------------#
lasso_reg3 = MyLassoRegression(alpha=100)
lasso_reg3.fit(x, y)
