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
from scipy.stats import t  # Import t-distribution for statistical tests

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

zi_t = np.zeros((n_stocks, n_months, n_characteristics * n_macro_factors)) 
for j in range(n_months):
    for i in range(n_stocks):
        zi_t[i, j] = np.outer(macro_factors[j],stock_characteristics[i, j]).flatten() 

theta = np.random.rand(n_characteristics * n_macro_factors)
ri_t_plus_1 = np.zeros((n_stocks, n_months))
for j in range(n_months):
    for i in range(n_stocks):
        ri_t_plus_1[i, j] = zi_t[i, j].dot(theta) + np.random.normal(0, 0.05)

zi_t_flattened = zi_t.reshape(n_stocks * n_months, -1)
ri_t_flattened = ri_t_plus_1.flatten()

zi_t_df = pd.DataFrame(zi_t_flattened, columns=[f"z_{k+1}" for k in range(zi_t_flattened.shape[1])])
zi_t_df["Stock_ID"] = np.repeat(range(1, n_stocks + 1), n_months)
zi_t_df["Month"] = np.tile(range(1, n_months + 1), n_stocks)

ri_t_df = pd.DataFrame({
    "Stock_ID": np.repeat(range(1, n_stocks + 1), n_months),
    "Month": np.tile(range(1, n_months + 1), n_stocks),
    "Excess_Return": ri_t_flattened,
    "Weights": np.random.rand(n_stocks * n_months)
})

data = pd.concat([zi_t_df, ri_t_df[["Excess_Return", "Weights"]]], axis=1)

# -------------------------------
# Part 2: Train-Test Split
# -------------------------------
X_full = data[[col for col in data.columns if col.startswith('z_')]]
Y_full = data['Excess_Return']
# TODO: Split zi_t_flattened and ri_t_flattened into a training-validation set and a test set (with test set proportion of 0.3)
X_trainval, X_test, Y_trainval, Y_test = train_test_split(
    X_full, Y_full, test_size=0.3, random_state=42
)

# TODO: Further divide the training-validation set into a training set and a validation set (with a validation set proportion of approximately 0.2857)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_trainval, Y_trainval, test_size=0.2857, random_state=42
)

# -------------------------------
# Part 3: Model Training Section
# -------------------------------

# --- OLS Regression ---
ols_model = LinearRegression()
ols_model.fit(X_trainval, Y_trainval)

# --- Weighted Linear Regression ---
weighted_ols_model = LinearRegression()
weighted_ols_model.fit(X_trainval, Y_trainval, sample_weight=np.random.rand(len(Y_trainval)))

# --- Huber Regressor ---
huber_model = HuberRegressor()
huber_model.fit(X_trainval, Y_trainval)

# --- ElasticNet Model Tuning ---
best_mse = float('inf')
best_alpha = None
best_l1_ratio = None

for alpha in [0.01, 0.1, 1.0, 10.0]:
    for l1_ratio in [0.1, 0.5, 0.9]:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=1000, random_state=42)
        model.fit(X_train, Y_train)
        y_val_pred = model.predict(X_val)
        val_mse = mean_squared_error(Y_val, y_val_pred)

        if val_mse < best_mse:
            best_mse = val_mse
            best_alpha = alpha
            best_l1_ratio = l1_ratio

elastic_net_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, fit_intercept=True, max_iter=1000, random_state=42)
elastic_net_model.fit(X_trainval, Y_trainval)

# --- Principal Component Regression (PCR) ---
best_pcr_mse = float('inf')
best_n_components_pca = None

for n_components in range(1, min(X_train.shape[1], 20) + 1):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    lr_pcr = LinearRegression()
    lr_pcr.fit(X_train_pca, Y_train)
    
    y_val_pred = lr_pcr.predict(X_val_pca)
    val_mse = mean_squared_error(Y_val, y_val_pred)

    if val_mse < best_pcr_mse:
        best_pcr_mse = val_mse
        best_n_components_pca = n_components

pca_model = PCA(n_components=best_n_components_pca)
X_trainval_pca = pca_model.fit_transform(X_trainval)
lr_pcr_model = LinearRegression()
lr_pcr_model.fit(X_trainval_pca, Y_trainval)

# --- Partial Least Squares Regression (PLS) ---
best_pls_mse = float('inf')
best_n_components_pls = None

for n_components in range(1, min(X_train.shape[1], 20) + 1):
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, Y_train)
    
    y_val_pred = pls.predict(X_val)
    val_mse = mean_squared_error(Y_val, y_val_pred)

    if val_mse < best_pls_mse:
        best_pls_mse = val_mse
        best_n_components_pls = n_components

pls_model = PLSRegression(n_components=best_n_components_pls)
pls_model.fit(X_trainval, Y_trainval)

# --- Generalized Linear Model (Spline Transformation + ElasticNet) ---
best_val_mse = float('inf')
best_alpha = None
best_l1_ratio = None

spline_transformer = SplineTransformer(n_knots=5, degree=2, include_bias=False, knots="uniform")
X_train_spline = spline_transformer.fit_transform(X_train)
X_val_spline = spline_transformer.transform(X_val)

for l1_ratio in [0.1, 0.5, 0.9]:
    for alpha in np.logspace(-3, 1, 10):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=1000, random_state=42)
        model.fit(X_train_spline, Y_train)
        
        y_val_pred = model.predict(X_val_spline)
        val_mse = mean_squared_error(Y_val, y_val_pred)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_l1_ratio = l1_ratio
            best_alpha = alpha

glm_spline_transformer = SplineTransformer(n_knots=5, degree=2, include_bias=False, knots="uniform")
X_trainval_spline = glm_spline_transformer.fit_transform(X_trainval)

glm_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, fit_intercept=True, max_iter=1000, random_state=42)
glm_model.fit(X_trainval_spline, Y_trainval)

# --- non-linear models ---

# --- Neural Network Model ---
def create_model(learning_rate=0.01, neurons_layer1=32, neurons_layer2=16, neurons_layer3=8):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(zi_t_flattened.shape[1],)),
        tf.keras.layers.Dense(neurons_layer1, activation='relu'),
        tf.keras.layers.Dense(neurons_layer2, activation='relu'),
        tf.keras.layers.Dense(neurons_layer3, activation='relu'),
        tf.keras.layers.Dense(1) 
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

best_model = None
best_val_loss = float('inf')
learning_rates = [0.001, 0.01, 0.1]
neurons_layer1_options = [32, 64]
neurons_layer2_options = [16, 32]
neurons_layer3_options = [8, 16]

for lr in learning_rates:
    for n1 in neurons_layer1_options:
        for n2 in neurons_layer2_options:
            for n3 in neurons_layer3_options:
                model = create_model(learning_rate=lr, 
                                     neurons_layer1=n1, 
                                     neurons_layer2=n2, 
                                     neurons_layer3=n3)
                
                history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50, batch_size=32, verbose=0, callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                ])
                
                val_loss = min(history.history['val_loss'])
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model

nn_model = best_model

# --- Gradient Boosting Regressor ---
brt_best_params = None
brt_best_model = None
brt_best_rmse = float('inf')

for n_estimators in [50, 100, 200]:
    for learning_rate in [0.01, 0.1, 0.2]:
        for max_depth in [3, 5, 7]:
            model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
            model.fit(X_train, Y_train)
            val_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(Y_val, val_pred))

            if val_rmse < brt_best_rmse:
                brt_best_rmse = val_rmse
                brt_best_params = {"n_estimators":n_estimators, "learning_rate": learning_rate, "max_depth": max_depth}
                brt_best_model = model

brt_model = brt_best_model

# --- Random Forest Regressor ---
rf_best_params = None
rf_best_model = None
rf_best_rmse = float('inf')

for n_estimators in [50, 100, 200]:
    for max_depth in [3, 5, 7]:
        for min_samples_split in [2, 5, 10]:
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            model.fit(X_train, Y_train)
            val_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(Y_val, val_pred))

            if val_rmse < rf_best_rmse:
                rf_best_rmse = val_rmse
                rf_best_params = {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_split": min_samples_split}
                rf_best_model = model

rf_model = rf_best_model

# -------------------------------
# Part 4: Prediction Wrappers
# -------------------------------
def predict_all_models(X):
    return {
        "OLS": ols_model.predict(X),
        "Weighted OLS": weighted_ols_model.predict(X),
        "Huber": huber_model.predict(X),
        "ElasticNet": elastic_net_model.predict(X),
        "PCR": lr_pcr_model.predict(pca_model.transform(X)),
        "PLS": pls_model.predict(X).flatten(),
        "GLM": glm_model.predict(glm_spline_transformer.transform(X)),
        "Neural Network": nn_model.predict(X).flatten(),
        "Boosted Regression Trees": brt_model.predict(X),
        "Random Forest": rf_model.predict(X)
    }

prediction = predict_all_models(X_test)

# -------------------------------
# Part 5: Full-Sample Time Series Plots - to see the predictions vs. actuals
# -------------------------------
prediciton_full_sample = predict_all_models(X_full)

for model, preds in prediciton_full_sample.items():
    preds_reshape = preds.reshape(n_stocks, n_months)
    y_reshape = Y_full.values.reshape(n_stocks, n_months)

    n_cols = 2
    n_rows = int(np.ceil(n_stocks / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i in range(n_stocks):
            ax = axes[i]
            ax.plot(y_reshape[i], label='Actual', color='blue', marker='o')
            ax.plot(preds_reshape[i], label=f"{model} Predicted", color='red', linestyle='--', marker='x')
            ax.set_title(f'Stock {i+1}')
            ax.set_xlabel('Months')
            ax.set_ylabel('Returns')
            ax.legend()
            ax.grid(False)

    for j in range(n_stocks, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f"images/prediction/{model}_full_sample_prediction_comparison.png")


# -------------------------------
# Part 6: Out-of-Sample R² Results Table - to evaluate model performance
# -------------------------------
def calc_r2(Y_true, Y_pred):
    ss_res = np.sum((Y_true - Y_pred) ** 2)
    ss_tot = np.sum(Y_true** 2)
    return 1 - (ss_res / ss_tot)

r2_test_results = {name: calc_r2(Y_test, pred) for name, pred in prediction.items()}
r2_test_df = pd.DataFrame.from_dict(r2_test_results, orient='index', columns=['R²']).to_csv("data/r2_results.csv", index=True)

# -------------------------------
# Part 7: Diebold-Mariano Test Statistics - to compare model predictions
# -------------------------------
def diebold_mariano_test(Y_true, pred_1, pred_2):
    """
    Perform the Diebold-Mariano test for comparing two predictive models.
    
    Parameters:
    Y_true : array-like
        True values of the dependent variable.
    pred_1 : array-like
        Predictions from the first model.
    pred_2 : array-like
        Predictions from the second model.
    
    Returns:
    dm_statistic : float
        Diebold-Mariano test statistic.
    p_value : float
        p-value of the Diebold-Mariano test.
    """
    e1 = Y_true - pred_1
    e2 = Y_true - pred_2
    diff = e1 ** 2 - e2 ** 2
    dm_statistic = np.mean(diff) / np.sqrt(np.var(diff) / len(Y_true))
    p_value = 2 * (1 - t.cdf(np.abs(dm_statistic), df=len(diff)-1))
    return dm_statistic, p_value

def compare_models(Y_true, predictions):
    df_diebold_mariano = pd.DataFrame(columns=['Model1', 'Model2', 'DM Statistic', 'p-value'])
    models = list(predictions.keys())
    for i in range(len(models)):
        for j in range(len(models)):
            model1 = models[i]
            model2 = models[j]
            dm_statistic, p_value = diebold_mariano_test(Y_true, predictions[model1], predictions[model2])
            df_diebold_mariano = df_diebold_mariano._append({
                'Model1': model1,
                'Model2': model2,
                'DM Statistic': dm_statistic,
                'p-value': p_value
            }, ignore_index=True)
    return df_diebold_mariano

df_dm = compare_models(Y_test, prediction)
df_dm.to_csv("data/diebold_mariano_results.csv", index=False)

# Show the results in a heatmap

# Pivot to matrix form
heatmap_matrix = np.empty((len(df_dm['Model1'].unique()), len(df_dm['Model2'].unique())), dtype=object)
model_names = df_dm['Model1'].unique()

for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        if i == j:
            heatmap_matrix[i, j] = 'white'
            continue
        row = df_dm[(df_dm['Model1'] == m1) & (df_dm['Model2'] == m2)]
        pval = row['p-value'].values[0]
        dmstat = row['DM Statistic'].values[0]
        if pval > 0.05:
            heatmap_matrix[i, j] = 'white'
        elif pval <= 0.05 and dmstat < 0:
            heatmap_matrix[i, j] = 'green'
        else:
            heatmap_matrix[i, j] = 'red'

# Convert color matrix to numeric for plotting
color_map = {'white': 0, 'green': 1, 'red': 2}
numeric_matrix = np.vectorize(color_map.get)(heatmap_matrix)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_matrix, annot=False, cmap=sns.color_palette(['white', 'green', 'red']), 
            xticklabels=model_names, yticklabels=model_names, cbar=False)
plt.title('Diebold-Mariano Test Heatmap')
plt.xlabel('Model 2')
plt.ylabel('Model 1')
plt.tight_layout()
plt.savefig("images/diebold/diebold_mariano_heatmap.png")

# -------------------------------
# Part 8: Variable Importance Calculations & Heatmaps - to understand feature importance ( to see which features are more important)
# -------------------------------
def variable_importance(X, Y, prediction):
    base_r2 = {name: calc_r2(Y, pred) for name, pred in prediction.items()}
    importance = {}
    for i in range(X.shape[1]):
        X_zerod = X.copy()
        X_zerod.iloc[:, i] = 0
        pred_drop = predict_all_models(X_zerod)
        r2_drop = {name: calc_r2(Y, pred) for name, pred in pred_drop.items()}
        importance[f'Var{i+1}'] = {name: base_r2[name] - r2_drop[name] for name in base_r2}
    return pd.DataFrame(importance).T

importance_df = variable_importance(X_test, Y_test, prediction)
importance_df.to_csv("data/variable_importance_results.csv", index=True)

sns.heatmap(importance_df, annot=True, cmap='viridis', xticklabels=importance_df.columns, yticklabels=importance_df.index)
plt.title('Variable Importance (Drop in R²) for All Models')
plt.tight_layout()
plt.savefig("images/variable_importance/variable_importance_all_models.png")
plt.close()

for col in importance_df.columns:
    plt.figure(figsize=(10, 6))
    sns.heatmap(importance_df[[col]].T, annot=True, cmap="viridis", cbar=True, fmt=".4f")
    plt.title(f'Variable Importance for {col}')
    plt.xlabel('Variables')
    plt.ylabel('Drop in R²')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"images/variable_importance/variable_importance_{col}.png")
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
    
    plt.title(f"Prediction vs. actual Return for each decil for {name}")
    plt.xlabel("Decile")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"images/decile/pred_vs_actual_{name}.png")

for name in list(prediction.keys()):
    df_decile = decile_portfolio_analysis(Y_test, prediction[name])
    df_decile.to_csv(f"data/decile_analysis_{name}.csv")
    plot_pred_vs_actual(df_decile, name)