# multiple_linear_regression_from_scratch.py

#--------------------------------------------------#
# Creating Class for Multiple Linear Regression    #
#--------------------------------------------------#
class MyLinearRegression:
    
    def __init__(self):
        # Initialize slope (coefficients) and intercept
        self.slope = None
        self.intercept = None

    def fit(self, x_train, y_train):
        # Add bias column (x0 = 1) to the feature matrix
        x_train = np.insert(x_train, 0, 1, axis=1)
        
        # Calculate regression coefficients using
        # Normal Equation: β = (XᵀX)⁻¹ Xᵀ y
        betas = np.linalg.inv(
            np.dot(x_train.T, x_train)
        ).dot(x_train.T).dot(y_train)
        
        # First coefficient is intercept
        self.intercept = betas[0]
        
        # Remaining coefficients are slopes for each feature
        self.slope = betas[1:]
    
    def predict(self, x_test):
        # Predict target values using learned coefficients
        y_pred = np.dot(x_test, self.slope) + self.intercept
        return y_pred


#--------------------------------------------------#
# Importing the Libraries                           #
#--------------------------------------------------#
import numpy as np


#--------------------------------------------------#
# Importing the Dataset                             #
#--------------------------------------------------#
from sklearn.datasets import load_diabetes

# Load diabetes dataset (features and target)
x, y = load_diabetes(return_X_y=True)


#--------------------------------------------------#
# Splitting the data into train and test data       #
#--------------------------------------------------#
from sklearn.model_selection import train_test_split

# 75% training data, 25% testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=10
)

# Shape of training data (samples, features)
x_train.shape


#--------------------------------------------------#
# Importing self-created algorithm class            #
#--------------------------------------------------#
regressor = MyLinearRegression()

# Train the custom multiple linear regression model
regressor.fit(x_train, y_train)


#--------------------------------------------------#
# Predicting test data results                      #
#--------------------------------------------------#
# Predict a single test instance
regressor.predict(x_test[4])

# Predict all test samples
y_pred = regressor.predict(x_test)


#--------------------------------------------------#
# Evaluating our model accuracy                    #
#--------------------------------------------------#
from sklearn.metrics import r2_score

# Calculate R² score
score = r2_score(y_test, y_pred)
score


#--------------------------------------------------#
# Checking calculated values from our class         #
#--------------------------------------------------#
# Learned feature coefficients (slopes)
regressor.slope

# Learned intercept
regressor.intercept
