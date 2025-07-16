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
from scipy.stats import t # Import t-distribution for statistical tests

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
plt.savefig("images/ReturnTimeSeriesAll.png")'''

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
pcr = LinearRegression()
pcr.fit(X_train_pca, Y_train)

# --- Partial Least Squares Regression (PLS) ---

pls = PLSRegression(n_components=5)
pls.fit(X_train, Y_train)


# --- Generalized Linear Model (Spline Transformation + ElasticNet) ---

spline = SplineTransformer(n_knots=4, degree=3)
X_train_spline = spline.fit_transform(X_train[:, [0]])
X_train_glm = np.hstack([X_train_spline, X_train[:, 1:]])
glm = ElasticNet(alpha=0.1, l1_ratio=0.5)
glm.fit(X_train_glm, Y_train)

# --- non-linear models ---

# --- Neural Network Model ---

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_val, Y_val))

# --- Gradient Boosting Regressor ---

gbr = GradientBoostingRegressor()
gbr.fit(X_train, Y_train)

# --- Random Forest Regressor ---

rf = RandomForestRegressor()
rf.fit(X_train, Y_train)


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

prediction = predict_all_models(X_test)

# -------------------------------
# Part 5: Full-Sample Time Series Plots - to see the predictions vs. actuals
# -------------------------------

# plot timeseries for each model
def plot_timeseries_per_model(Y_true, models):
    n_models = len(models)
    n_cols = 2
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    axes = axes.flatten()
    for idx, (name, pred) in enumerate(models.items()):
        ax = axes[idx]
        ax.plot(Y_true, label='Actual')
        ax.plot(pred, label='Predicted')
        ax.set_title(f'{name} Predictions vs Actuals')
        ax.legend()
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig("images/TSPerModel.png")
    plt.close()

#plot_timeseries_per_model(Y_test, prediction)

# -------------------------------
# Part 6: Out-of-Sample R² Results Table - to evaluate model performance
# -------------------------------
# TODO: Calculate R² according to the formula: 1 - (sum of squared errors / total sum of squares)

def calc_r2(Y_true, Y_pred):
     ss_res = np.sum((Y_true - Y_pred) ** 2)
     ss_tot = np.sum((Y_true - np.mean(Y_true)) ** 2)
     return 1 - (ss_res / ss_tot)

r2_results = {name: calc_r2(Y_test, pred) for name, pred in prediction.items()}
df_r2 = pd.DataFrame.from_dict(r2_results, orient='index', columns=['R²']).to_csv("data/r2_results.csv")

# -------------------------------
# Part 7: Diebold-Mariano Test Statistics - to compare model predictions
# -------------------------------

def diebold_mariano_test(Y_true, pred1, pred2):
    e1 = Y_true - pred1
    e2 = Y_true - pred2
    diff = e1 ** 2 - e2 ** 2
    dm_statistic = np.mean(diff) / np.sqrt(np.var(diff) / len(Y_true))
    p_value = 2 * (1 - t.cdf(np.abs(dm_statistic), df=len(diff)-1))
    return dm_statistic, p_value

def compare_models(Y_true, predictions):
    df_diebold_mariano = pd.DataFrame(columns=['Model1', 'Model2', 'DM Statistic', 'p-value'])
    model_names = list(predictions.keys())
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            dm_statistic, p_value = diebold_mariano_test(Y_true, predictions[model1], predictions[model2])
            df_diebold_mariano = df_diebold_mariano._append({
                'Model1': model1,
                'Model2': model2,
                'DM Statistic': dm_statistic,
                'p-value': p_value
            }, ignore_index=True)
    return df_diebold_mariano

df_diebold_mariano = compare_models(Y_test, prediction).to_csv("data/diebold_mariano_results.csv", index=False)

# Build a heatmap based on Diebold-Mariano results

# Load results from CSV
df_dm = pd.read_csv("data/diebold_mariano_results.csv")

# Pivot to matrix form
heatmap_matrix = np.empty((len(df_dm['Model1'].unique()), len(df_dm['Model2'].unique())), dtype=object)
model_names = df_dm['Model1'].unique()

for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        row = df_dm[(df_dm['Model1'] == m1) & (df_dm['Model2'] == m2)]
        pval = row['p-value'].values[0]
        dmstat = row['DM Statistic'].values[0]
        if pval > 0.05:
            heatmap_matrix[i, j] = 'yellow'
        elif pval <= 0.05 and dmstat < 0:
            heatmap_matrix[i, j] = 'green'
        else:
            heatmap_matrix[i, j] = 'red'

# Convert color matrix to numeric for plotting
color_map = {'yellow': 0, 'green': 1, 'red': 2}
numeric_matrix = np.vectorize(color_map.get)(heatmap_matrix)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_matrix, annot=False, cmap=sns.color_palette(['yellow', 'green', 'red']), 
            xticklabels=model_names, yticklabels=model_names, cbar=False)
plt.title('Diebold-Mariano Test Heatmap')
plt.xlabel('Model 2')
plt.ylabel('Model 1')
plt.tight_layout()
plt.savefig("images/diebold_mariano_heatmap.png")


# -------------------------------
# Part 8: Variable Importance Calculations & Heatmaps - to understand feature importance ( to see which features are more important)
# -------------------------------
# TODO: Define a function to compute variable importance based on the drop in R² when a feature is removed

def variable_importance_all_models(X, Y):
    """
    For each variable, remove it from X, run predict_all_models, and save the drop in R² for each model.
    Returns a DataFrame with shape (n_features, n_models) containing the drop in R².
    """
    base_pred = predict_all_models(X)
    base_r2 = {name: calc_r2(Y, pred) for name, pred in base_pred.items()}
    importances = {}
    for i in range(X.shape[1]):
        X_zerod = X.copy()
        X_zerod[:, i] = 0
        pred_drop = predict_all_models(X_zerod)
        r2_drop = {name: calc_r2(Y, pred) for name, pred in pred_drop.items()}
        # Importance: drop in R² when feature i is removed
        importances[f'Var{i+1}'] = {name: base_r2[name] - r2_drop[name] for name in base_r2}
    return pd.DataFrame(importances).T  # rows: variables, columns: models

# Calculate and plot variable importance for all models
imp_all = variable_importance_all_models(X_test, Y_test)
imp_all.to_csv("data/variable_importance_all_models.csv")
sns.heatmap(imp_all, annot=True, cmap='viridis', xticklabels=imp_all.columns, yticklabels=imp_all.index)
plt.title('Variable Importance (Drop in R²) for All Models')
plt.tight_layout()
plt.savefig("images/variable_importance_all_models.png")

for col in imp_all.columns:
    plt.figure(figsize=(10, 6))
    sns.heatmap(imp_all[[col]].T, annot=True, cmap="viridis", cbar=True, fmt=".4f")
    plt.title(f'Variable Importance for {col}')
    plt.xlabel('Variables')
    plt.ylabel('Drop in R²')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/variable_importance_{col}.png")
    plt.close()

    
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

def plot_pred_vs_actual(results_df, name):
    plt.figure(figsize=(10, 6))
    actual = results_df['Actual']['mean']
    predicted = results_df['Predicted']['mean']
    
    plt.plot(actual.index, actual.values, marker='o', label='Tatsächliche Rendite')
    plt.plot(predicted.index, predicted.values, marker='x', label='Vorhergesagte Rendite')
    
    plt.title(f"Vorhergesagte vs. tatsächliche Renditen pro Dezil für {name}")
    plt.xlabel("Dezil (nach Prognose sortiert)")
    plt.ylabel("Durchschnittliche Rendite")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"images/pred_vs_actual_{name}.png")


for name in list(prediction.keys()):
    df_decile = decile_portfolio_analysis(Y_test, prediction[name])
    df_decile.to_csv(f"data/decile_analysis_{name}.csv")
    plot_pred_vs_actual(df_decile, name)



# -------------------------------


tut_pred = predict_all_models(X_full)

ols_tut_pred = tut_pred['OLS']

ols_tut_pred_reshape = ols_tut_pred.reshape(n_stocks, n_months)
y_reshape = Y_full.reshape(n_stocks, n_months)


def plot_ols_preds_per_stock(y_reshape, ols_tut_pred_reshape, n_stocks=10):
    n_cols = 2
    n_rows = int(np.ceil(n_stocks / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(n_stocks):
        ax = axes[i]
        ax.plot(y_reshape[i], label='Actual', color='blue', marker='o')
        ax.plot(ols_tut_pred_reshape[i], label='OLS Predicted', color='red', linestyle='--', marker='x')
        ax.set_title(f'Stock {i+1}')
        ax.set_xlabel('Months')
        ax.set_ylabel('Returns')
        ax.legend()
        ax.grid(False)
    for j in range(n_stocks, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("images/ols_preds_all_stocks.png")


plot_ols_preds_per_stock(y_reshape, ols_tut_pred_reshape, n_stocks=n_stocks)
