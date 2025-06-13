import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from statsmodels.formula.api import ols # For ANOVA in Python
import statsmodels.api as sm # For OLS summary
import warnings
import os
from datetime import datetime

# Suppress specific warnings that might arise during processing
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# For plotting multiple plots into a single PDF
from matplotlib.backends.backend_pdf import PdfPages

# ================================#
## RF, XGBoost, and GBM PREDICTIVE CODES ##
# ================================#

## Load the dataset
try:
    spotify_charts_2024 = pd.read_csv("universal_top_spotify_songs.new.csv")
    # If the above fails, try a full path
except FileNotFoundError:
    print("CSV file not found in current directory. Please provide the full path.")


### --------------------------------
## DATA CLEANING
### -------------------------------

# market counts for each song
spotify_charts_2024['market_count'] = spotify_charts_2024.groupby('spotify_id')['country'].transform('nunique')

# remove all duplicates, pick the most popular song(country)
# Similar to slice_max, sort by popularity and drop_duplicates keeping the first
spotify_charts_2024['other_charted_countries'] = spotify_charts_2024.groupby('spotify_id')['country'].transform(lambda x: ', '.join(x.unique()))
spotify_charts_2024 = spotify_charts_2024.sort_values(by='popularity', ascending=False).drop_duplicates(subset=['spotify_id'], keep='first').reset_index(drop=True)


# Function to count the number of artists in each row
spotify_charts_2024['artist_count'] = spotify_charts_2024['artists'].apply(lambda x: len(str(x).split(',')))

# Convert character date columns to Date objects using mdy()
spotify_charts_2024['snapshot_date'] = pd.to_datetime(spotify_charts_2024['snapshot_date'])
spotify_charts_2024['album_release_date'] = pd.to_datetime(spotify_charts_2024['album_release_date'])
spotify_charts_2024['days_out'] = (spotify_charts_2024['snapshot_date'] - spotify_charts_2024['album_release_date']).dt.days

# change boolean into integers
spotify_charts_2024['is_explicit'] = spotify_charts_2024['is_explicit'].astype(int)

# Standardize duration_ms (convert to minutes)
spotify_charts_2024['duration_min'] = spotify_charts_2024['duration_ms'] / 60000
spotify_charts_2024 = spotify_charts_2024.drop(columns=['duration_ms']) # Remove original column

# Remove unneeded columns
columns_to_drop = [
    'country', 'other_charted_countries', 'snapshot_date', 'name',
    'artists', 'album_name', 'album_release_date', 'spotify_id'
]
spotify_modelling = spotify_charts_2024.drop(columns=columns_to_drop, errors='ignore')

# check for missing values
print("\nMissing values before handling:")
print(spotify_modelling.isnull().sum())

# Handle missing values by imputing with mean (for numerical columns)
# For categorical columns, you'd typically use mode or a constant, but given the R code's
# `mutate_all(~ifelse(is.na(.), mean(., na.rm = TRUE), .))` and the `select` later,
# it implies all remaining columns are numeric or will be treated as such for imputation.
for col in spotify_modelling.columns:
    if spotify_modelling[col].dtype in ['int64', 'float64']:
        spotify_modelling[col].fillna(spotify_modelling[col].mean(), inplace=True)
    # else: For non-numeric, you'd use fillna(spotify_modelling[col].mode()[0], inplace=True) or similar
print("\nMissing values after handling:")
print(spotify_modelling.isnull().sum())

# Arrange the columns - Python DataFrames maintain column order
# This step is important to match the R code's column order for operations like ggpairs
ordered_columns = [
    'popularity', 'days_out', 'artist_count', 'market_count', 'daily_rank',
    'daily_movement', 'weekly_movement', 'duration_min', 'is_explicit', 'mode',
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'key', 'time_signature'
]
spotify_modelling = spotify_modelling[ordered_columns]


# Remove popularity 0
spotify_modelling = spotify_modelling[spotify_modelling['popularity'] != 0].sort_values(by='popularity', ascending=False)


# ------------------------------
### DESCRIPTIVE STATISTICS
# ------------------------------------

# Function to compute basic statistics
def spotify_stats(column):
    stats_dict = {
        'Mean': column.mean(),
        'Median': column.median(),
        'SD': column.std(),
        'Variance': column.var(),
        'IQR': column.quantile(0.75) - column.quantile(0.25)
    }
    return pd.Series(stats_dict)

# Loop through columns and compute statistics
stats_results = spotify_modelling.apply(spotify_stats).T # .T for transpose to match R's output structure

print("\nDescriptive Statistics:")
print(stats_results)

# Start the PDF device for plots
pdf_filename = "J-K-spotify_analysis_all_models_plots.pdf"
pdf_pages = PdfPages(pdf_filename)

# Plot 1: Popularity to daily_rank
plot_cols_1 = spotify_modelling.columns[0:5] # Columns 1 to 5 (0-indexed in Python)
fig1 = sns.pairplot(spotify_modelling[plot_cols_1])
fig1.fig.suptitle('Scatter Plot Matrix 1 (Popularity to Daily Rank)', y=1.02) # Adjust title position
pdf_pages.savefig(fig1.fig) # Save the figure to PDF
plt.close(fig1.fig) # Close the figure to free memory

# Plot 2: daily_movement to mode
plot_cols_2 = spotify_modelling.columns[5:10] # Columns 6 to 10
fig2 = sns.pairplot(spotify_modelling[plot_cols_2])
fig2.fig.suptitle('Scatter Plot Matrix 2 (Daily Movement to Mode)', y=1.02)
pdf_pages.savefig(fig2.fig)
plt.close(fig2.fig)

# Plot 3: danceability to instrumentalness
plot_cols_3 = spotify_modelling.columns[10:15] # Columns 11 to 15
fig3 = sns.pairplot(spotify_modelling[plot_cols_3])
fig3.fig.suptitle('Scatter Plot Matrix 3 (Danceability to Instrumentalness)', y=1.02)
pdf_pages.savefig(fig3.fig)
plt.close(fig3.fig)

# Plot 4: liveness to time_signature
plot_cols_4 = spotify_modelling.columns[15:21] # Columns 16 to 21
fig4 = sns.pairplot(spotify_modelling[plot_cols_4])
fig4.fig.suptitle('Scatter Plot Matrix 4 (Liveness to Time Signature)', y=1.02)
pdf_pages.savefig(fig4.fig)
plt.close(fig4.fig)

# check popularity distribution
plt.figure(figsize=(10, 6))
sns.histplot(spotify_modelling['popularity'], bins=50, kde=True, color='skyblue')
plt.title('Popularity Distribution After Cleaning')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
pdf_pages.savefig()
plt.close() # Close the figure

# ----------------------------
### CORRELATION
# -------------------------
# Select the audio feature columns
audio_features = spotify_modelling[[
    'popularity', 'market_count', 'daily_rank', 'daily_movement', 'weekly_movement',
    'days_out', 'artist_count', 'duration_min', 'is_explicit', 'mode',
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'time_signature', 'liveness', 'valence', 'key', 'tempo'
]]
# Compute correlation matrix
correlation_matrix = audio_features.corr(numeric_only=True)

# Plot heatmap with correlation values
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Heatmap", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
pdf_pages.savefig()
plt.close()

# -------------------------
### ANOVA FOR VARIANCE IMPORTANCE
# ---------------------------

# Convert relevant columns to factors in Python (object/category dtype for categorical)
# We'll make a copy to avoid modifying the original spotify_modelling for other parts
spotify_modelling_anova = spotify_modelling.copy()
categorical_cols = ['mode', 'is_explicit', 'key', 'time_signature']
for col in categorical_cols:
    if col in spotify_modelling_anova.columns:
        spotify_modelling_anova[col] = spotify_modelling_anova[col].astype('category')

anova_results = []
for col in spotify_modelling_anova.columns:
    if col == 'popularity':
        continue # Skip target variable

    formula = f'popularity ~ C({col})' if spotify_modelling_anova[col].dtype == 'category' else f'popularity ~ {col}'
    try:
        model = ols(formula, data=spotify_modelling_anova).fit()
        anova_table = sm.stats.anova_lm(model, typ=2) # Type 2 ANOVA for unbalanced designs
        f_value = anova_table['F'][0]
        p_value = anova_table['PR(>F)'][0]
        anova_results.append({'Feature': col, 'F_Value': f_value, 'P_Value': p_value})
    except Exception as e:
        print(f"Could not perform ANOVA/LM for column {col}: {e}")
        anova_results.append({'Feature': col, 'F_Value': np.nan, 'P_Value': np.nan})

# Create a data frame for variance importance based on ANOVA
anova_importance = pd.DataFrame(anova_results)
anova_importance.dropna(subset=['F_Value'], inplace=True) # Remove rows where F_Value is NA
anova_importance = anova_importance.sort_values(by='F_Value', ascending=False).reset_index(drop=True)

print("\nANOVA Variance Importance (Combined):")
print(anova_importance)

# --- Plotting ANOVA results ---

# 1. Bar plot of p-values
plt.figure(figsize=(12, 7))
sns.barplot(x='Feature', y='P_Value', data=anova_importance, palette='viridis')
plt.axhline(y=0.05, color='red', linestyle='--', label='Significance (0.05)')
plt.title('P-values from ANOVA and Linear Regression')
plt.xlabel('Feature')
plt.ylabel('P-value')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
pdf_pages.savefig()
plt.close()

# 2. Scatter plot of F-values vs. p-values
plt.figure(figsize=(12, 7))
sns.scatterplot(x='F_Value', y='P_Value', data=anova_importance, hue='Feature', s=100, palette='tab10')
for i, row in anova_importance.iterrows():
    plt.text(row['F_Value'] + 0.1, row['P_Value'], row['Feature'], fontsize=8)
plt.axhline(y=0.05, color='red', linestyle='--', label='Significance (0.05)')
plt.title('F-values vs. P-values')
plt.xlabel('F-value')
plt.ylabel('P-value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
pdf_pages.savefig()
plt.close()


# --------------------------
### SKLEARN DATA PARTITION
# ---------------------------

# Define feature columns and target variable
X = spotify_modelling.drop(columns=['popularity'])
y = spotify_modelling['popularity']

# Split the data into training and testing sets
# set.seed(50) in R is equivalent to random_state=50 in Python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, stratify=y if y.nunique() < 10 else None) # Stratify if target is categorical-like

# Converting to DataFrame is not strictly necessary for scikit-learn,
# but done to mimic R's explicit conversion.
train_dframe_features = X_train.copy()
test_dframe_features = X_test.copy()
train_dframe_target = y_train.to_frame(name='y_train')
test_dframe_target = y_test.to_frame(name='y_test')

# Combine features and target for distribution plotting, if needed
train_dframe = pd.concat([train_dframe_features, train_dframe_target], axis=1)
test_dframe = pd.concat([test_dframe_features, test_dframe_target], axis=1)


# testing popularity distribution for training data
plt.figure(figsize=(10, 6))
sns.histplot(y_train, bins=50, kde=True, color='skyblue')
plt.title('Popularity Distribution for Training Data')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
pdf_pages.savefig()
plt.close()

# testing popularity distribution for testing data
plt.figure(figsize=(10, 6))
sns.histplot(y_test, bins=50, kde=True, color='skyblue')
plt.title('Popularity Distribution for Testing Data')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
pdf_pages.savefig()
plt.close()

# ------------------------------------
## DATA PREPARATION FOR MODELING
# ------------------------------------

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(include='object').columns.tolist() # Ensure 'mode', 'is_explicit', 'key', 'time_signature' were not converted to category dtype prematurely if they contain strings

# Re-check categorical columns, as they might have been converted to int/float earlier
# based on the R script's `as.integer` for is_explicit and `as.factor` for others.
# In Python, we want true categorical columns for OneHotEncoder.
# Assuming 'mode', 'key', 'time_signature' are originally integers acting as categories
# and 'is_explicit' is int 0/1. Let's explicitly define them based on original intent.
categorical_cols_for_ohe = [col for col in ['is_explicit', 'mode', 'key', 'time_signature'] if col in X_train.columns]
numerical_cols_for_scaling = [col for col in numerical_cols if col not in categorical_cols_for_ohe]


# Create a preprocessing pipeline
# `step_center` and `step_scale` are `StandardScaler` in Python.
# `step_dummy` is `OneHotEncoder`.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols_for_scaling),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols_for_ohe)
    ],
    remainder='passthrough' # Keep other columns not specified (if any)
)

# Apply preprocessing
# The `fit_transform` and `transform` methods handle fitting on training and transforming both.
train_processed_array = preprocessor.fit_transform(X_train)
test_processed_array = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out()

# Convert back to DataFrame
train_processed = pd.DataFrame(train_processed_array, columns=feature_names, index=X_train.index)
test_processed = pd.DataFrame(test_processed_array, columns=feature_names, index=X_test.index)


print("\nProcessed Training Data (First few rows):\n")
print(train_processed.head())
print("\nProcessed Testing Data (First few rows):\n")
print(test_processed.head())

# Combine processed features with the target variable (for modeling libraries that prefer a single DataFrame)
# Note: For scikit-learn models, X_train_processed and y_train are often passed separately.
# This step is more for mimicking R's `cbind` structure.
train_processed_combined = pd.concat([train_processed, y_train.reset_index(drop=True)], axis=1) # Reset index to align
test_processed_combined = pd.concat([test_processed, y_test.reset_index(drop=True)], axis=1)
train_processed_combined.rename(columns={y_train.name: 'y_train'}, inplace=True)
test_processed_combined.rename(columns={y_test.name: 'y_test'}, inplace=True)

# -------------------------
### RANDOM FOREST MODEL
# --------------------------

# Hyperparameter tuning using GridSearchCV/RandomizedSearchCV (caret::train equivalent)
# RandomForestRegressor from sklearn doesn't directly expose `mtry` (n_features_to_consider)
# but `max_features` is the equivalent. `min.node.size` is `min_samples_leaf`.
# `splitrule` is implicitly handled by the algorithm (usually 'mse' or 'squared_error').
# `num.trees` is `n_estimators`.

from sklearn.model_selection import GridSearchCV

# Define the model
rf_model = RandomForestRegressor(random_state=50, n_jobs=-1) # n_jobs=-1 uses all available cores

# Define the hyperparameter grid
# mtry (features selected at each split) -> max_features
# min.node.size -> min_samples_leaf
# num.trees -> n_estimators
param_grid_rf = {
    'n_estimators': [200], # Corresponds to num.trees = 200
    'max_features': [0.25, 0.35, 0.45], # Rough equivalents for mtry = c(5, 7, 9, 11) for ~20 features (5/20=0.25, 7/20=0.35, etc.)
    'min_samples_leaf': [3, 5, 7] # Corresponds to min.node.size
}
# Adjusted max_features based on approximate number of features after one-hot encoding
# Assuming original features ~20, if 4 are categorical, and 2 of those have 12 keys, 2 modes, 3 time_signatures
# original features: 20
# Numerical: 20 - 4 = 16
# Categorical: is_explicit (2), mode (2), key (12), time_signature (3) -> (2-1) + (2-1) + (12-1) + (3-1) = 1 + 1 + 11 + 2 = 15 dummy variables
# Total features: 16 (numerical) + 15 (dummy) = 31 features.
# So, mtry 5, 7, 9, 11 correspond to max_features roughly 5/31=0.16, 7/31=0.22, 9/31=0.29, 11/31=0.35.
# Let's adjust param_grid_rf based on this estimation.

# A more robust way to define max_features is as a fraction of total features
num_features = train_processed.shape[1]
# Example: mtry=5 becomes max_features=5
# If R's mtry is absolute number of features, then we can use int values for max_features
param_grid_rf = {
    'n_estimators': [200],
    'max_features': [5, 7, 9, 11], # Directly use the values from R's mtry
    'min_samples_leaf': [3, 5, 7]
}


# GridSearchCV for hyperparameter tuning (equivalent to caret::train with 'cv')
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf,
                              cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=0)
grid_search_rf.fit(train_processed, y_train)

rf_best_model = grid_search_rf.best_estimator_

# Model results
print("\nRandom Forest - Best Parameters:", grid_search_rf.best_params_)
print("Random Forest - Best Cross-Validation RMSE:", np.sqrt(-grid_search_rf.best_score_))

### Save RF model
import joblib
joblib.dump(rf_best_model, "rf_model.pkl")

# Predictions on test set
rf_pred = rf_best_model.predict(test_processed)

# Evaluate model performance
MAE_rf = mean_absolute_error(y_test, rf_pred)
RMSE_rf = np.sqrt(mean_squared_error(y_test, rf_pred))
R_squared_rf = r2_score(y_test, rf_pred)

print("Random Forest - Mean Absolute Error (MAE):", MAE_rf)
print("Random Forest - Root Mean Squared Error (RMSE):", RMSE_rf)
print("Random Forest - R-squared (R²):", R_squared_rf)

# Feature Importance
importance_df_rf = pd.DataFrame({
    'Feature': train_processed.columns,
    'Importance': rf_best_model.feature_importances_
})
sorted_importance_df_rf = importance_df_rf.sort_values(by='Importance', ascending=False).reset_index(drop=True)
print("\nRandom Forest - Feature Importance:")
print(sorted_importance_df_rf)

# -------------------------
### XGBOOST MODEL
# --------------------------

# Prepare data for XGBoost (already done by preprocessor and X_train/X_test)
# XGBoost models in scikit-learn handle DataFrames directly.

# Define the model
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=50, n_jobs=-1)

# Define the hyperparameter grid
# nrounds -> n_estimators
# max_depth, eta (learning_rate), gamma, colsample_bytree, min_child_weight, subsample are direct equivalents
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1], # eta
    'gamma': [0],
    'colsample_bytree': [0.7, 0.9],
    'min_child_weight': [1],
    'subsample': [0.8]
}

# GridSearchCV for hyperparameter tuning
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb,
                               cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=0)
grid_search_xgb.fit(train_processed, y_train)

xgb_best_model = grid_search_xgb.best_estimator_

# Model results
print("\nXGBoost - Best Parameters:", grid_search_xgb.best_params_)
print("XGBoost - Best Cross-Validation RMSE:", np.sqrt(-grid_search_xgb.best_score_))

### Save XGBoost model
joblib.dump(xgb_best_model, "xgb_model.pkl")

# Predictions on test set
xgb_pred = xgb_best_model.predict(test_processed)

# Evaluate model performance
MAE_xgb = mean_absolute_error(y_test, xgb_pred)
RMSE_xgb = np.sqrt(mean_squared_error(y_test, xgb_pred))
R_squared_xgb = r2_score(y_test, xgb_pred)

print("XGBoost - Mean Absolute Error (MAE):", MAE_xgb)
print("XGBoost - Root Mean Squared Error (RMSE):", RMSE_xgb)
print("XGBoost - R-squared (R²):", R_squared_xgb)

# Feature Importance
importance_df_xgb = pd.DataFrame({
    'Feature': train_processed.columns,
    'Importance': xgb_best_model.feature_importances_
})
sorted_importance_df_xgb = importance_df_xgb.sort_values(by='Importance', ascending=False).reset_index(drop=True)
print("\nXGBoost - Feature Importance:")
print(sorted_importance_df_xgb)

# -------------------------
### GBM MODEL
# --------------------------

# Define the model
gbm_model = GradientBoostingRegressor(random_state=50)

# Define the hyperparameter grid
# n.trees -> n_estimators
# interaction.depth -> max_depth
# shrinkage -> learning_rate
# n.minobsinnode -> min_samples_leaf
param_grid_gbm = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05],
    'min_samples_leaf': [10, 20]
}

# GridSearchCV for hyperparameter tuning
grid_search_gbm = GridSearchCV(estimator=gbm_model, param_grid=param_grid_gbm,
                               cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=0)
grid_search_gbm.fit(train_processed, y_train)

gbm_best_model = grid_search_gbm.best_estimator_

# Model results
print("\nGBM - Best Parameters:", grid_search_gbm.best_params_)
print("GBM - Best Cross-Validation RMSE:", np.sqrt(-grid_search_gbm.best_score_))

### Save GBM model
joblib.dump(gbm_best_model, "gbm_model.pkl")

# Predictions on test set
gbm_pred = gbm_best_model.predict(test_processed)

# Evaluate model performance
MAE_gbm = mean_absolute_error(y_test, gbm_pred)
RMSE_gbm = np.sqrt(mean_squared_error(y_test, gbm_pred))
R_squared_gbm = r2_score(y_test, gbm_pred)

print("Gradient Boosting Machine - Mean Absolute Error (MAE):", MAE_gbm)
print("Gradient Boosting Machine - Root Mean Squared Error (RMSE):", RMSE_gbm)
print("Gradient Boosting Machine - R-squared (R²):", R_squared_gbm)

# Feature Importance
importance_df_gbm = pd.DataFrame({
    'Feature': train_processed.columns,
    'Importance': gbm_best_model.feature_importances_
})
sorted_importance_df_gbm = importance_df_gbm.sort_values(by='Importance', ascending=False).reset_index(drop=True)
print("\nGBM - Feature Importance:")
print(sorted_importance_df_gbm)

# -----------------------
## MODEL COMPARISON:
# -------------------------

# Create a data frame to compare model performance
model_comparison = pd.DataFrame({
    'Model': ["Random Forest", "XGBoost", "GBM"],
    'MAE': [MAE_rf, MAE_xgb, MAE_gbm],
    'RMSE': [RMSE_rf, RMSE_xgb, RMSE_gbm],
    'R_squared': [R_squared_rf, R_squared_xgb, R_squared_gbm]
})

print("\nModel Performance Comparison:")
print(model_comparison)

# Create a bar plot for R-squared comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R_squared', data=model_comparison, palette='viridis')
for index, row in model_comparison.iterrows():
    plt.text(index, row['R_squared'], round(row['R_squared'], 3), color='black', ha="center", va='bottom')
plt.title("R-squared Comparison of Models")
plt.ylabel("R-squared")
plt.tight_layout()
pdf_pages.savefig()
plt.close()

# Create a bar plot for RMSE comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', data=model_comparison, palette='viridis')
for index, row in model_comparison.iterrows():
    plt.text(index, row['RMSE'], round(row['RMSE'], 3), color='black', ha="center", va='bottom')
plt.title("RMSE Comparison of Models")
plt.ylabel("RMSE")
plt.tight_layout()
pdf_pages.savefig()
plt.close()

# Create a bar plot for MAE comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MAE', data=model_comparison, palette='viridis')
for index, row in model_comparison.iterrows():
    plt.text(index, row['MAE'], round(row['MAE'], 3), color='black', ha="center", va='bottom')
plt.title("MAE Comparison of Models")
plt.ylabel("MAE")
plt.tight_layout()
pdf_pages.savefig()
plt.close()

# ------------------------------------
## ACTUAL VS PREDICTED PLOTS:
# ------------------------------------

# Random Forest
plt.figure(figsize=(8, 8))
plot_data_rf = pd.DataFrame({'Actual': y_test, 'Predicted': rf_pred})
sns.scatterplot(x='Actual', y='Predicted', data=plot_data_rf, color='blue', alpha=0.6)
sns.regplot(x='Actual', y='Predicted', data=plot_data_rf, scatter=False, color='red', line_kws={'linestyle': '--'})
plt.title("Actual vs Predicted Popularity (Random Forest)")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.grid(True, linestyle='--', alpha=0.5)
pdf_pages.savefig()
plt.close()

# XGBoost
plt.figure(figsize=(8, 8))
plot_data_xgb = pd.DataFrame({'Actual': y_test, 'Predicted': xgb_pred})
sns.scatterplot(x='Actual', y='Predicted', data=plot_data_xgb, color='green', alpha=0.6)
sns.regplot(x='Actual', y='Predicted', data=plot_data_xgb, scatter=False, color='red', line_kws={'linestyle': '--'})
plt.title("Actual vs Predicted Popularity (XGBoost)")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.grid(True, linestyle='--', alpha=0.5)
pdf_pages.savefig()
plt.close()

# GBM
plt.figure(figsize=(8, 8))
plot_data_gbm = pd.DataFrame({'Actual': y_test, 'Predicted': gbm_pred})
sns.scatterplot(x='Actual', y='Predicted', data=plot_data_gbm, color='purple', alpha=0.6)
sns.regplot(x='Actual', y='Predicted', data=plot_data_gbm, scatter=False, color='red', line_kws={'linestyle': '--'})
plt.title("Actual vs Predicted Popularity (GBM)")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.grid(True, linestyle='--', alpha=0.5)
pdf_pages.savefig()
plt.close()

# Stop the PDF device - this is crucial to save the PDF
pdf_pages.close()

print(f"\nAll plots, including the model comparison and actual vs predicted plots, have been saved to {pdf_filename} in your working directory.")