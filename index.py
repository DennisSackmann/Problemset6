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

print(stock_characteristics)

# Generate true coefficients for characteristics and macro factors
true_betas = np.random.randn(n_characteristics)
true_gammas = np.random.randn(n_macro_factors)

# Generate returns: ri_t = z_i_t @ beta + x_t @ gamma + noise
noise = np.random.normal(0, 0.5, (n_stocks, n_months))
ri_t = np.zeros((n_stocks, n_months))
for i in range(n_stocks):
    for t in range(n_months):
        ri_t[i, t] = (
            stock_characteristics[i, t, :] @ true_betas
            + macro_factors[t, :] @ true_gammas
            + noise[i, t]
        )

# Flatten data for regression (stack stocks and months)
zi_t_flattened = stock_characteristics.reshape(-1, n_characteristics)
xi_t_flattened = np.repeat(macro_factors, n_stocks, axis=0)
ri_t_flattened = ri_t.flatten()
X_full = np.hstack([zi_t_flattened, xi_t_flattened])
Y_full = ri_t_flattened



# -------------------------------
# Part 2: Train-Test Split
# -------------------------------
# TODO: Split zi_t_flattened and ri_t_flattened into a training-validation set and a test set (with test set proportion of 0.3)

# TODO: Further divide the training-validation set into a training set and a validation set (with a validation set proportion of approximately 0.2857)

# Split into train-validation and test sets
X_trainval, X_test, Y_trainval, Y_test = train_test_split(
    X_full, Y_full, test_size=0.3, random_state=42
)
# Further split train-validation into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(
    X_trainval, Y_trainval, test_size=0.2857, random_state=42
)

# -------------------------------
# Part 3: Model Training Section
# -------------------------------

# --- OLS Regression ---

# OLS Regression
print("Training OLS Regression")
ols = LinearRegression()
ols.fit(X_train, Y_train)
Y_pred_ols = ols.predict(X_test)

# --- Weighted Linear Regression ---

# Weighted Linear Regression (using random weights for demonstration)
print("Training Weighted OLS Regression")
weights = np.random.rand(len(Y_train))
ols_weighted = LinearRegression()
ols_weighted.fit(X_train, Y_train, sample_weight=weights)
Y_pred_ols_weighted = ols_weighted.predict(X_test)

# --- Huber Regressor ---

print("Training Huber Regressor")
huber = HuberRegressor()
huber.fit(X_train, Y_train)
Y_pred_huber = huber.predict(X_test)


# --- ElasticNet Model Tuning ---
print("Training ElasticNet Regression")
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, Y_train)
Y_pred_elastic = elastic.predict(X_test)
           

# --- Principal Component Regression (PCR) ---

# PCR: PCA + Linear Regression
print("Training PCR")
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
pcr = LinearRegression()
pcr.fit(X_train_pca, Y_train)
Y_pred_pcr = pcr.predict(X_test_pca)


# --- Partial Least Squares Regression (PLS) ---
print("Training PLS Regression")
pls = PLSRegression(n_components=5)
pls.fit(X_train, Y_train)
Y_pred_pls = pls.predict(X_test).flatten()



# --- Generalized Linear Model (Spline Transformation + ElasticNet) ---

# Spline transformation on first characteristic
print("Training Generalized Linear Model with Spline Transformation")
spline = SplineTransformer(n_knots=4, degree=3)
X_train_spline = spline.fit_transform(X_train[:, [0]])
X_test_spline = spline.transform(X_test[:, [0]])
X_train_glm = np.hstack([X_train_spline, X_train[:, 1:]])
X_test_glm = np.hstack([X_test_spline, X_test[:, 1:]])
glm = ElasticNet(alpha=0.1, l1_ratio=0.5)
glm.fit(X_train_glm, Y_train)
Y_pred_glm = glm.predict(X_test_glm)

# --- non-linear models ---

# Neural Network Model
print("Training Neural Network Model")
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_val, Y_val))
Y_pred_nn = nn_model.predict(X_test).flatten()

# Gradient Boosting Regressor
print("Training Gradient Boosting Regressor")
gbr = GradientBoostingRegressor()
gbr.fit(X_train, Y_train)
Y_pred_gbr = gbr.predict(X_test)

# Random Forest Regressor
print("Training Random Forest Regressor")
rf = RandomForestRegressor()
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)

# --- Neural Network Model ---

# --- Gradient Boosting Regressor ---

# --- Random Forest Regressor ---


# -------------------------------
# Part 4: Prediction Wrappers
# -------------------------------

def predict_all_models(X):
    return {
        'OLS': ols.predict(X),
        'Weighted OLS': ols_weighted.predict(X),
        'Huber': huber.predict(X),
        'ElasticNet': elastic.predict(X),
        'PCR': pcr.predict(pca.transform(X)),
        'PLS': pls.predict(X).flatten(),
        'GLM': glm.predict(np.hstack([spline.transform(X[:, [0]]), X[:, 1:]])),
        'NN': nn_model.predict(X).flatten(),
        'GBR': gbr.predict(X),
        'RF': rf.predict(X)
    }

# -------------------------------
# Part 5: Full-Sample Time Series Plots - to see the predictions vs. actuals
# -------------------------------


def plot_all_predictions(Y_true, models):
    n_models = len(models)
    n_cols = 2
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()
    for idx, (name, pred) in enumerate(models.items()):
        ax = axes[idx]
        ax.plot(Y_true, label='Actual')
        ax.plot(pred, label='Predicted')
        print(pred.shape)
        ax.set_title(f'{name} Predictions vs Actuals')
        ax.legend()
    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

# Example: plot OLS predictions

# Plot predictions vs actuals for every model

print("Plotting all predictions vs actuals in one figure")
models = predict_all_models(X_test)
plot_all_predictions(Y_test, models)

# -------------------------------
# Part 6: Out-of-Sample R² Results Table - to evaluate model performance
# -------------------------------
# TODO: Calculate R² according to the formula: 1 - (sum of squared errors / total sum of squares)

def calc_r2(Y_true, Y_pred):
    ss_res = np.sum((Y_true - Y_pred) ** 2)
    ss_tot = np.sum((Y_true - np.mean(Y_true)) ** 2)
    return 1 - ss_res / ss_tot

print("Calculating Out-of-Sample R² for all models")
models = predict_all_models(X_test)
r2_results = {name: calc_r2(Y_test, pred) for name, pred in models.items()}
print('Out-of-sample R²:')
for name, r2 in r2_results.items():
    print(f'{name}: {r2:.3f}')

# -------------------------------
# Part 7: Diebold-Mariano Test Statistics - to compare model predictions
# -------------------------------

def diebold_mariano_test(Y_true, pred1, pred2):
    e1 = Y_true - pred1
    e2 = Y_true - pred2
    diff = e1 ** 2 - e2 ** 2
    mean_diff = np.mean(diff)
    var_diff = np.var(diff) / len(diff)
    dm_stat = mean_diff / np.sqrt(var_diff)
    return dm_stat

# Example: DM test between OLS and NN
print("Calculating Diebold-Mariano Test Statistic for OLS vs NN")
dm_stat = diebold_mariano_test(Y_test, Y_pred_ols, Y_pred_nn)
print(f'Diebold-Mariano statistic (OLS vs NN): {dm_stat:.3f}')
# -------------------------------
# Part 8: Variable Importance Calculations & Heatmaps - to understand feature importance ( to see which features are more important)
# -------------------------------
# TODO: Define a function to compute variable importance based on the drop in R² when a feature is removed

def variable_importance(model, X, Y):
    base_r2 = calc_r2(Y, model.predict(X))
    importances = []
    for i in range(X.shape[1]):
        X_drop = np.delete(X, i, axis=1)
        # Retrain model for fair comparison
        m = LinearRegression()
        m.fit(X_drop, Y)
        r2_drop = calc_r2(Y, m.predict(X_drop))
        importances.append(base_r2 - r2_drop)
    return importances

print("Calculating Variable Importance for OLS")
imp_ols = variable_importance(ols, X_test, Y_test)
sns.heatmap(np.array(imp_ols).reshape(1, -1), annot=True, cmap='viridis', xticklabels=[f'Feat{i+1}' for i in range(X_test.shape[1])])
plt.title('OLS Variable Importance (Drop in R²)')
plt.show()
# -------------------------------
# -------------------------------
# Part 9: Auxiliary Functions and Decile Portfolio Analysis - to analyze model performance across deciles - to compare predicted vs actual  sharpe ratios
# -------------------------------

def decile_portfolio_analysis(Y_true, Y_pred):
    df = pd.DataFrame({'Actual': Y_true, 'Predicted': Y_pred})
    df['Decile'] = pd.qcut(df['Predicted'], 10, labels=False)
    results = df.groupby('Decile').agg({'Actual': ['mean', 'std'], 'Predicted': ['mean', 'std']})
    results['Sharpe_Actual'] = results['Actual']['mean'] / results['Actual']['std']
    results['Sharpe_Predicted'] = results['Predicted']['mean'] / results['Predicted']['std']
    return results

# Example: decile analysis for OLS
print("Performing Decile Portfolio Analysis for OLS")
print(decile_portfolio_analysis(Y_test, Y_pred_ols))
