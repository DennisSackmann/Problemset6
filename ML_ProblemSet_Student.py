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

ols = LinearRegression()
ols.fit(X_train, Y_train)

# --- Weighted Linear Regression ---

weights = np.random.rand(len(Y_train))
ols_weighted = LinearRegression()
ols_weighted.fit(X_train, Y_train, sample_weight=weights)   

# --- Huber Regressor ---

huber = HuberRegressor()
huber.fit(X_train, Y_train)

# --- ElasticNet Model Tuning ---

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, Y_train)           

# --- Principal Component Regression (PCR) ---

pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
pcr = LinearRegression()
pcr.fit(X_train_pca, Y_train)
Y_pred_pcr = pcr.predict(X_test_pca)

# --- Partial Least Squares Regression (PLS) ---

pls = PLSRegression(n_components=5)
pls.fit(X_train, Y_train)


# --- Generalized Linear Model (Spline Transformation + ElasticNet) ---

spline = SplineTransformer(n_knots=4, degree=3)
X_train_spline = spline.fit_transform(X_train[:, [0]])
X_test_spline = spline.transform(X_test[:, [0]])
X_train_glm = np.hstack([X_train_spline, X_train[:, 1:]])
X_test_glm = np.hstack([X_test_spline, X_test[:, 1:]])
glm = ElasticNet(alpha=0.1, l1_ratio=0.5)
glm.fit(X_train_glm, Y_train)
Y_pred_glm = glm.predict(X_test_glm)

# --- non-linear models ---

# --- Neural Network Model ---

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_val, Y_val))
Y_pred_nn = nn_model.predict(X_test).flatten()

# --- Gradient Boosting Regressor ---

gbr = GradientBoostingRegressor()
gbr.fit(X_train, Y_train)
Y_pred_gbr = gbr.predict(X_test)

# --- Random Forest Regressor ---

rf = RandomForestRegressor()
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)


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
