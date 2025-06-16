print("import basic libraries")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("import Sci-kit Learn modules")
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor 

print("import SciPy and Statsmodels for statistical analysis (ANOVA, etc.)")
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

print("import System and Utility modules")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import joblib
print("-"*20)
print("Loading the dataset...")
print("-"*20)
try:
    spotify_charts_2024 = pd.read_csv("universal_top_spotify_songs.new.csv")
    # If the above fails, try a full path
except FileNotFoundError:
    print("CSV file not found in current directory. Please provide the full path.")



print("-"*20)
print("DATA CLEANING")
print("-"*20)

print("adding market counts for each song")
spotify_charts_2024['market_count'] = spotify_charts_2024.groupby('spotify_id')['country'].transform('nunique')

print("Check for non-string types in 'country' column, Before removing non-string rows:")
print(spotify_charts_2024[['spotify_id', 'country']].head())
print(spotify_charts_2024['country'].apply(type).value_counts())

print("Removing rows where 'country' is not a string, ")
print("We also check for pd.NA which is a nullable type for object columns in newer pandas")
spotify_charts_2024 = spotify_charts_2024[
    spotify_charts_2024['country'].apply(lambda x: isinstance(x, str) or pd.isna(x))
].copy()

print("Converting 'country' column to string type...")
spotify_charts_2024['country'] = spotify_charts_2024['country'].astype(str)

print("After removing non-string rows and converting to string:")
print(spotify_charts_2024[['spotify_id', 'country']].head())
print(spotify_charts_2024['country'].apply(type).value_counts())

print("Checking for duplicate rows based on 'spotify_id' and 'country'...")
spotify_charts_2024['other_charted_countries'] = spotify_charts_2024.groupby('spotify_id')['country'].transform(
    lambda x: ', '.join(x.unique())
)
spotify_charts_2024 = spotify_charts_2024.sort_values(by='popularity', ascending=False).drop_duplicates(subset=['spotify_id'],
 keep='first').reset_index(drop=True)

print("count te no. of artists in each song")
spotify_charts_2024['artist_count'] = spotify_charts_2024['artists'].apply(lambda x: len(str(x).split(',')))

print("converting date columns to datetime format and creating  a new column days out.")
spotify_charts_2024['snapshot_date'] = pd.to_datetime(spotify_charts_2024['snapshot_date'])
spotify_charts_2024['album_release_date'] = pd.to_datetime(spotify_charts_2024['album_release_date'])
spotify_charts_2024['days_out'] = (spotify_charts_2024['snapshot_date'] - spotify_charts_2024['album_release_date']).dt.days

print("change boolean into integers")
spotify_charts_2024['is_explicit'] = spotify_charts_2024['is_explicit'].astype(int)

print("Standardize duration_ms (convert to minutes)")
spotify_charts_2024['duration_min'] = spotify_charts_2024['duration_ms'] / 60000
spotify_charts_2024 = spotify_charts_2024.drop(columns=['duration_ms']) # Remove original column

print("Removing unneeded columns")
columns_to_drop = [
    'country', 'other_charted_countries', 'snapshot_date', 'name',
    'artists', 'album_name', 'album_release_date', 'spotify_id'
]
spotify_modelling = spotify_charts_2024.drop(columns=columns_to_drop, errors='ignore')

print("Missing values before handling:")
print(spotify_modelling.isnull().sum())

print("Handle missing values by imputing with mean (for numerical columns)")
print("For categorical columns, you'd typically use mode or a constant")
for col in spotify_modelling.columns:
    if spotify_modelling[col].dtype in ['int64', 'float64']:
        spotify_modelling[col].fillna(spotify_modelling[col].mean(), inplace=True)
    else:
       spotify_modelling[col].fillna(spotify_modelling[col].mode()[0], inplace=True)

print("Missing values after handling:")
print(spotify_modelling.isnull().sum())

print("Arrange the columns ")
ordered_columns = [
    'popularity', 'days_out', 'artist_count', 'market_count', 'daily_rank',
    'daily_movement', 'weekly_movement', 'duration_min', 'is_explicit', 'mode',
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'key', 'time_signature'
]
spotify_modelling = spotify_modelling[ordered_columns]


print("Remove popularity 0")
spotify_modelling = spotify_modelling[spotify_modelling['popularity'] != 0].sort_values(by='popularity', ascending=False)
print("-"*20)
print("DESCRIPTIVE STATISTICS")
print("-"*20)

print("Function to compute basic statistics")
def spotify_stats(column):
    stats_dict = {
        'Mean': column.mean(),
        'Median': column.median(),
        'SD': column.std(),
        'Variance': column.var(),
        'IQR': column.quantile(0.75) - column.quantile(0.25)
    }
    return pd.Series(stats_dict)

print("Looping through columns and compute statistics")
stats_results = spotify_modelling.apply(spotify_stats).T

print("Descriptive Statistics:  scatter plots of inter-features")
print(stats_results)

print("Plot 1: Popularity to daily_rank")
plot_cols_1 = spotify_modelling.columns[0:5]
fig1 = sns.pairplot(spotify_modelling[plot_cols_1])
fig1.fig.suptitle('Scatter Plot Matrix 1 (Popularity to Daily Rank)', y=1.02)
plt.show()
plt.close(fig1.fig)

print("Plot 2: daily_movement to mode")
plot_cols_2 = spotify_modelling.columns[5:10]
fig2 = sns.pairplot(spotify_modelling[plot_cols_2])
fig2.fig.suptitle('Scatter Plot Matrix 2 (Daily Movement to Mode)', y=1.02)
plt.show()
plt.close(fig2.fig)

print("Plot 3: danceability to instrumentalness")
plot_cols_3 = spotify_modelling.columns[10:15]
fig3 = sns.pairplot(spotify_modelling[plot_cols_3])
fig3.fig.suptitle('Scatter Plot Matrix 3 (Danceability to Instrumentalness)', y=1.02)
plt.show()
plt.close(fig3.fig)

print("Plot 4: liveness to time_signature")
plot_cols_4 = spotify_modelling.columns[15:21]
fig4 = sns.pairplot(spotify_modelling[plot_cols_4])
fig4.fig.suptitle('Scatter Plot Matrix 4 (Liveness to Time Signature)', y=1.02)
plt.show()
plt.close(fig4.fig)

print("check popularity distribution")
plt.figure(figsize=(10, 6))
sns.histplot(spotify_modelling['popularity'], bins=50, kde=True, color='skyblue')
plt.title('Popularity Distribution After Cleaning')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()
plt.close()

print("-"*20)
print("CORRELATION")
print("-"*20)

print("Select the audio feature columns for correlation analysis")
audio_features = spotify_modelling[[
    'popularity', 'market_count', 'daily_rank', 'daily_movement', 'weekly_movement',
    'days_out', 'artist_count', 'duration_min', 'is_explicit', 'mode',
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'time_signature', 'liveness', 'valence', 'key', 'tempo'
]]
print("Compute correlation matrix")
correlation_matrix = audio_features.corr(numeric_only=True)

print("Plot heatmap with correlation values")
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Heatmap", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
plt.close()

print("-"*20)
print("ANOVA FOR FEATURE IMPORTANCE")
print("-"*20)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
print("Performing ANOVA for feature importance, check categorical and continous variables")
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

print("Create a data frame for variance importance based on ANOVA")
anova_importance = pd.DataFrame(anova_results)
anova_importance.dropna(subset=['F_Value'], inplace=True) # Remove rows where F_Value is NA
anova_importance = anova_importance.sort_values(by='F_Value', ascending=False).reset_index(drop=True)

print("ANOVA Feature Importance (Combined):")
print(anova_importance)


print("1. Bar plot of p-values")
plt.figure(figsize=(12, 7))
sns.barplot(x='Feature', y='P_Value', data=anova_importance, palette='viridis')
plt.axhline(y=0.05, color='red', linestyle='--', label='Significance (0.05)')
plt.title('P-values from ANOVA and Linear Regression')
plt.xlabel('Feature')
plt.ylabel('P-value')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

print("2. Scatter plot of F-values vs. p-values")
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
plt.show()
plt.close()
print("-"*20)
print("SKLEARN DATA PARTITION")
print("-"*20)

print("Define feature columns and target variable")
X = spotify_modelling.drop(columns=['popularity'])
y = spotify_modelling['popularity']

print("Split the data into training and testing sets and set random_state for reproducibility")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, stratify=y if y.nunique() < 10 else None)

print("create a data frame of the sets")
train_dframe_features = X_train.copy()
test_dframe_features = X_test.copy()
train_dframe_target = y_train.to_frame(name='y_train')
test_dframe_target = y_test.to_frame(name='y_test')

print("combinning features and target into data frames")
train_dframe = pd.concat([train_dframe_features, train_dframe_target], axis=1)
test_dframe = pd.concat([test_dframe_features, test_dframe_target], axis=1)


print("testing popularity distribution for training data")
plt.figure(figsize=(10, 6))
sns.histplot(y_train, bins=50, kde=True, color='skyblue')
plt.title('Popularity Distribution for Training Data')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()
plt.close()

print("testing popularity distribution for testing data")
plt.figure(figsize=(10, 6))
sns.histplot(y_test, bins=50, kde=True, color='skyblue')
plt.title('Popularity Distribution for Testing Data')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()
plt.close()

print("-"*20)
print("DATA PREPARATION FOR MODELING")
print("-"*20)

print("Identify numerical and categorical columns")
numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

print("Re-check categorical columns, as they might have been converted to int/float earlier")
categorical_cols_for_ohe = [col for col in ['is_explicit', 'mode', 'key', 'time_signature'] if col in X_train.columns]
numerical_cols_for_scaling = [col for col in numerical_cols if col not in categorical_cols_for_ohe]


print("Create a preprocessing pipeline")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols_for_scaling),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols_for_ohe)
    ],
    remainder='passthrough'
)

print("Apply preprocessing")
train_processed_array = preprocessor.fit_transform(X_train)
test_processed_array = preprocessor.transform(X_test)

print("Get feature names after one-hot encoding")
feature_names = preprocessor.get_feature_names_out()

print("Convert back to DataFrame")
train_processed = pd.DataFrame(train_processed_array, columns=feature_names, index=X_train.index)
test_processed = pd.DataFrame(test_processed_array, columns=feature_names, index=X_test.index)


print("Processed Training Data (First few rows):")
print(train_processed.head())
print("Processed Testing Data (First few rows):")
print(test_processed.head())

print("Combine processed features with the target variable (for modeling libraries that prefer a single DataFrame)")
train_processed_combined = pd.concat([train_processed, y_train.reset_index(drop=True)], axis=1)
test_processed_combined = pd.concat([test_processed, y_test.reset_index(drop=True)], axis=1)
train_processed_combined.rename(columns={y_train.name: 'y_train'}, inplace=True)
test_processed_combined.rename(columns={y_test.name: 'y_test'}, inplace=True)
print("---" * 10)
print("RANDOM FOREST MODEL")
print("---" * 10)

print("**Hyperparameter tuning using GridSearchCV**")

print("Defining the Random Forest model and hyperparameter grid...")
rf_model = RandomForestRegressor(random_state=50, n_jobs=-1)
num_features = train_processed.shape[1]

param_grid_rf = {
    'n_estimators': [200],
    'max_features': [5, 7, 9, 11],
    'min_samples_leaf': [3, 5, 7]
}

print("Performing GridSearchCV for hyperparameter tuning...")
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf,
                              cv=5, n_jobs=-1, scoring='neg_mean_squared_error',
                              verbose=0)
grid_search_rf.fit(train_processed, y_train)

rf_best_model = grid_search_rf.best_estimator_

print("**Random Forest Model Results**")
print(f"  Best Parameters: {grid_search_rf.best_params_}")
print(f"  Best Cross-Validation RMSE: {np.sqrt(-grid_search_rf.best_score_):.4f}")

# Save RF model
joblib.dump(rf_best_model, "rf_model.pkl")
print("Random Forest model saved as 'rf_model.pkl'")

print("**Evaluating Random Forest on Test Set**")
rf_pred = rf_best_model.predict(test_processed)

MAE_rf = mean_absolute_error(y_test, rf_pred)
RMSE_rf = np.sqrt(mean_squared_error(y_test, rf_pred))
R_squared_rf = r2_score(y_test, rf_pred)

print(f"  Mean Absolute Error (MAE): {MAE_rf:.4f}")
print(f"  Root Mean Squared Error (RMSE): {RMSE_rf:.4f}")
print(f"  R-squared (R²): {R_squared_rf:.4f}")

print("**Random Forest - Top 10 Feature Importances**")
importance_df_rf = pd.DataFrame({
    'Feature': train_processed.columns,
    'Importance': rf_best_model.feature_importances_
})
sorted_importance_df_rf = importance_df_rf.sort_values(by='Importance',
ascending=False).reset_index(drop=True)
# Using to_markdown() for excellent table rendering in Quarto PDF/HTML
print(sorted_importance_df_rf[0:10].to_markdown(index=False))


# --- XGBOOST MODEL ---
print("---" * 10)
print("XGBOOST MODEL")
print("---" * 10)

print("**Hyperparameter tuning using GridSearchCV**")

print("Defining the XGBoost model and hyperparameter grid...")
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=50, n_jobs=-1)

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1], # eta
    'gamma': [0],
    'colsample_bytree': [0.7, 0.9],
    'min_child_weight': [1],
    'subsample': [0.8]
}

print("Performing GridSearchCV for hyperparameter tuning...")
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb,
                               cv=5, n_jobs=-1, scoring='neg_mean_squared_error',
                               verbose=0)
grid_search_xgb.fit(train_processed, y_train)

xgb_best_model = grid_search_xgb.best_estimator_

print("**XGBoost Model Results**")
print(f"  Best Parameters: {grid_search_xgb.best_params_}")
print(f"  Best Cross-Validation RMSE: {np.sqrt(-grid_search_xgb.best_score_):.4f}")

# Save XGBoost model
joblib.dump(xgb_best_model, "xgb_model.pkl")
print("XGBoost model saved as 'xgb_model.pkl'")

print("**Evaluating XGBoost on Test Set**")
xgb_pred = xgb_best_model.predict(test_processed)

MAE_xgb = mean_absolute_error(y_test, xgb_pred)
RMSE_xgb = np.sqrt(mean_squared_error(y_test, xgb_pred))
R_squared_xgb = r2_score(y_test, xgb_pred)

print(f"  Mean Absolute Error (MAE): {MAE_xgb:.4f}")
print(f"  Root Mean Squared Error (RMSE): {RMSE_xgb:.4f}")
print(f"  R-squared (R²): {R_squared_xgb:.4f}")

print("**XGBoost - Top 10 Feature Importances**")
importance_df_xgb = pd.DataFrame({
    'Feature': train_processed.columns,
    'Importance': xgb_best_model.feature_importances_
})
sorted_importance_df_xgb = importance_df_xgb.sort_values(by='Importance',
ascending=False).reset_index(drop=True)
print(sorted_importance_df_xgb[0:10].to_markdown(index=False))

# --- GBM MODEL ---
print("---" * 10)
print("GBM MODEL")
print("---" * 10)

print("**Hyperparameter tuning using GridSearchCV**")

print("Defining the GBM model and hyperparameter grid...")
gbm_model = GradientBoostingRegressor(random_state=50)

param_grid_gbm = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.025],
    'min_samples_leaf': [10, 20]
}

print("Performing GridSearchCV for hyperparameter tuning...")
grid_search_gbm = GridSearchCV(estimator=gbm_model, param_grid=param_grid_gbm,
                               cv=5, n_jobs=-1, scoring='neg_mean_squared_error',
                               verbose=0)
grid_search_gbm.fit(train_processed, y_train)

gbm_best_model = grid_search_gbm.best_estimator_

print("**GBM Model Results**")
print(f"  Best Parameters: {grid_search_gbm.best_params_}")
print(f"  Best Cross-Validation RMSE: {np.sqrt(-grid_search_gbm.best_score_):.4f}")

# Save GBM model
joblib.dump(gbm_best_model, "gbm_model.pkl")
print("GBM model saved as 'gbm_model.pkl'")

print("**Evaluating GBM on Test Set**")
gbm_pred = gbm_best_model.predict(test_processed)

MAE_gbm = mean_absolute_error(y_test, gbm_pred)
RMSE_gbm = np.sqrt(mean_squared_error(y_test, gbm_pred))
R_squared_gbm = r2_score(y_test, gbm_pred)

print(f"  Mean Absolute Error (MAE): {MAE_gbm:.4f}")
print(f"  Root Mean Squared Error (RMSE): {RMSE_gbm:.4f}")
print(f"  R-squared (R²): {R_squared_gbm:.4f}")

print("**GBM - Top 10 Feature Importances**")
importance_df_gbm = pd.DataFrame({
    'Feature': train_processed.columns,
    'Importance': gbm_best_model.feature_importances_
})
sorted_importance_df_gbm = importance_df_gbm.sort_values(by='Importance',
ascending=False).reset_index(drop=True)
print(sorted_importance_df_gbm[0:10].to_markdown(index=False)) # Use 0:10 for top 10

print("="*25)
print("All model training and evaluation complete.")
print("The models and their performance characteristics are printed above.")
print("="*25)
print("-"*20)
print("MODEL COMPARISON:")
print("-"*20)

print("Create a data frame to compare model performance")
model_comparison = pd.DataFrame({
    'Model': ["Random Forest", "XGBoost", "GBM"],
    'MAE': [MAE_rf, MAE_xgb, MAE_gbm],
    'RMSE': [RMSE_rf, RMSE_xgb, RMSE_gbm],
    'R_squared': [R_squared_rf, R_squared_xgb, R_squared_gbm]
})

print("Model Performance Comparison:")
print(model_comparison)

print("Create a bar plot for R-squared comparison")
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R_squared', data=model_comparison, palette='viridis')
for index, row in model_comparison.iterrows():
    plt.text(index, row['R_squared'], round(row['R_squared'], 3), color='black', ha="center", va='bottom')
plt.title("R-squared Comparison of Models")
plt.ylabel("R-squared")
plt.tight_layout()
plt.show()
plt.close()

print("Create a bar plot for RMSE comparison")
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', data=model_comparison, palette='viridis')
for index, row in model_comparison.iterrows():
    plt.text(index, row['RMSE'], round(row['RMSE'], 3), color='black', ha="center", va='bottom')
plt.title("RMSE Comparison of Models")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()
plt.close()

print("Create a bar plot for MAE comparison")
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MAE', data=model_comparison, palette='viridis')
for index, row in model_comparison.iterrows():
    plt.text(index, row['MAE'], round(row['MAE'], 3), color='black', ha="center", va='bottom')
plt.title("MAE Comparison of Models")
plt.ylabel("MAE")
plt.tight_layout()
plt.show()
plt.close()

print("-"*20)
print("ACTUAL VS PREDICTED PLOTS:")
print("-"*20)

print("1. Random Forest")
plt.figure(figsize=(8, 8))
plot_data_rf = pd.DataFrame({'Actual': y_test, 'Predicted': rf_pred})
sns.scatterplot(x='Actual', y='Predicted', data=plot_data_rf, color='blue', alpha=0.6)
sns.regplot(x='Actual', y='Predicted', data=plot_data_rf, scatter=False, color='red', line_kws={'linestyle': '--'})
plt.title("Actual vs Predicted Popularity (Random Forest)")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
plt.close()

print("2. XGBoost")
plt.figure(figsize=(8, 8))
plot_data_xgb = pd.DataFrame({'Actual': y_test, 'Predicted': xgb_pred})
sns.scatterplot(x='Actual', y='Predicted', data=plot_data_xgb, color='green', alpha=0.6)
sns.regplot(x='Actual', y='Predicted', data=plot_data_xgb, scatter=False, color='red', line_kws={'linestyle': '--'})
plt.title("Actual vs Predicted Popularity (XGBoost)")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
plt.close()

print("3. GBM")
plt.figure(figsize=(8, 8))
plot_data_gbm = pd.DataFrame({'Actual': y_test, 'Predicted': gbm_pred})
sns.scatterplot(x='Actual', y='Predicted', data=plot_data_gbm, color='purple', alpha=0.6)
sns.regplot(x='Actual', y='Predicted', data=plot_data_gbm, scatter=False, color='red', line_kws={'linestyle': '--'})
plt.title("Actual vs Predicted Popularity (GBM)")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
plt.close()