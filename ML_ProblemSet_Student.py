# Problem Set: Multi-Model Financial Prediction and Risk Analysis
# Student Exercise Manual
# Please complete the code

# -------------------------------
# Import necessary libraries
# -------------------------------
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import Pandas for data manipulation
import tensorflow as tf  # Import TensorFlow for building neural network models
import seaborn as sns  # Import Seaborn for statistical plotting
import matplotlib.pyplot as plt  # Import Matplotlib for creating plots
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting datasets
from sklearn.metrics import mean_squared_error, r2_score  # Import metrics for evaluating models
from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNet  # Import linear regression models
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
from sklearn.cross_decomposition import PLSRegression  # Import PLSRegression for partial least squares regression
from sklearn.preprocessing import SplineTransformer  # Import SplineTransformer for spline feature transformation
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # Import ensemble regression models

plt.style.use('seaborn-v0_8-whitegrid')  # Set the plotting style to Seaborn whitegrid
plt.rcParams['figure.figsize'] = (12, 8)  # Set default figure size for plots to 12x8 inches

# -------------------------------
# Part 1: Data Generation
# -------------------------------
n_stocks = 10  # Define the number of stocks
n_months = 24  # Define the number of months (time periods)
n_characteristics = 5  # Define the number of stock-specific characteristics
n_macro_factors = 3  # Define the number of macroeconomic factors

np.random.seed(42)  # Set the random seed for reproducibility
stock_characteristics = np.random.rand(n_stocks, n_months, n_characteristics)  # Generate random stock characteristic data
macro_factors = np.random.rand(n_months, n_macro_factors)  # Generate random macroeconomic factor data


# generate true coefficients
true__betas = np.random.randn(n_characteristics)
true_gammas = np.random.randn(n_macro_factors)

# generate returns
noise = np.random.normal(0, 0.5, (n_stocks, n_months))
ri_t = np.zeros((n_stocks, n_months))
for i in range(n_stocks):
    for j in range(n_months):
        ri_t[i, j] = (
            stock_characteristics[i, j, :] @ true__betas
            + macro_factors[j, :] @ true_gammas
            + noise[i, j]
        )

'''# plot the return series for every stock
n_rows = int(np.ceil(n_stocks / 2))
fig, axes = plt.subplots(n_rows, 2, figsize=(10, 2 * n_rows), sharex=True)
axes = axes.flatten()
for i in range(n_stocks):
    axes[i].plot(ri_t[i, :], label=f'Stock {i+1}')
    axes[i].set_ylabel('Return')
    axes[i].set_title(f'Return Series: Stock {i+1}')
    axes[i].legend()
for j in range(n_stocks, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.show()'''

# flatten data for modeling
zi_t_flattened = stock_characteristics.reshape(-1, n_characteristics)
xi_t_flattened = np.repeat(macro_factors, n_stocks, axis=0)
ri_t_flattened = ri_t.flatten()
X_full = np.hstack([zi_t_flattened, xi_t_flattened])
Y_full = ri_t_flattened


# -------------------------------
# Part 2: Train-Test Split
# -------------------------------
# TODO: Split zi_t_flattened and ri_t_flattened into a training-validation set and a test set (with test set proportion of 0.3)

# split into training-validation and test sets
X_trainval, X_test, Y_trainval, Y_test = train_test_split(
    X_full, Y_full, test_size=0.3, random_state=42
)

# TODO: Further divide the training-validation set into a training set and a validation set (with a validation set proportion of approximately 0.2857)

# split training-validation set into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(
    X_trainval, Y_trainval, test_size=0.2857, random_state=42
)


# -------------------------------
# Part 3: Model Training Section
# -------------------------------

# --- OLS Regression ---

# --- Weighted Linear Regression ---
   

# --- Huber Regressor ---


# --- ElasticNet Model Tuning ---
           

# --- Principal Component Regression (PCR) ---


# --- Partial Least Squares Regression (PLS) ---



# --- Generalized Linear Model (Spline Transformation + ElasticNet) ---

# --- non-linear models ---

# --- Neural Network Model ---

# --- Gradient Boosting Regressor ---

# --- Random Forest Regressor ---


# -------------------------------
# Part 4: Prediction Wrappers
# -------------------------------

# -------------------------------
# Part 5: Full-Sample Time Series Plots - to see the predictions vs. actuals
# -------------------------------

# -------------------------------
# Part 6: Out-of-Sample R² Results Table - to evaluate model performance
# -------------------------------
# TODO: Calculate R² according to the formula: 1 - (sum of squared errors / total sum of squares)

# -------------------------------
# Part 7: Diebold-Mariano Test Statistics - to compare model predictions
# -------------------------------
# -------------------------------
# Part 8: Variable Importance Calculations & Heatmaps - to understand feature importance ( to see which features are more important)
# -------------------------------
# TODO: Define a function to compute variable importance based on the drop in R² when a feature is removed
# -------------------------------
# -------------------------------
# Part 9: Auxiliary Functions and Decile Portfolio Analysis - to analyze model performance across deciles - to compare predicted vs actual  sharpe ratios
# -------------------------------
