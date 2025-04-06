#kiluva
#family>duty>honour

# Load the dataset
sportify_charts_2025 = pd.read_csv(r'C:\Users\HP ELITEBOOK 820 G4\OneDrive\Documents\DATA ANALYSIS & VISUALIZATION\Python\spotifyProject\universal_top_spotify_songs.new.csv')
print(sportify_charts_2025)
# Add a date difference column
sportify_charts_2025['days_out'] = (pd.to_datetime(sportify_charts_2025['snapshot_date']) -
                                    pd.to_datetime(sportify_charts_2025['album_release_date'])).dt.days
# Remove duplicate rows
sportify_charts_2025 = sportify_charts_2025.drop_duplicates(subset=['spotify_id'])

# Remove null and unnecessary columns
sportify_charts_2025.drop(['country', 'snapshot_date', 'name', 'artists',
                           'album_name', 'album_release_date','spotify_id'], axis=1, inplace=True)

# Convert boolean columns to integers
sportify_charts_2025['is_explicit'] = sportify_charts_2025['is_explicit'].astype(int)

# Change duration from milliseconds to minutes
sportify_charts_2025['duration_min'] = sportify_charts_2025['duration_ms'] / 60000
sportify_charts_2025.drop('duration_ms', axis=1, inplace=True)

# Handle missing values by replacing them with column means
sportify_charts_2025.fillna(sportify_charts_2025.mean(), inplace=True)

# Reorder and clean up unnecessary columns
columns_order = ['popularity', 'days_out', 'duration_min', 'daily_rank',
                 'daily_movement', 'weekly_movement', 'is_explicit', 'mode',
                 'time_signature', 'key', 'danceability', 'energy', 'loudness',
                 'liveness', 'tempo', 'instrumentalness', 'acousticness',
                 'speechiness', 'valence']
sportify_charts_2025 = sportify_charts_2025[columns_order]
sportify_charts_2025 = sportify_charts_2025.sort_values(by='popularity', ascending=False)
print(sportify_charts_2025)

# Correlation matrix visualization
numeric_features = sportify_charts_2025.select_dtypes(include=[np.number])
cor_matrix = numeric_features.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(cor_matrix, annot=False, cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix")
plt.show()

## XGBoost Model ##
# Split the dataset into training and testing sets
train_data, test_data = train_test_split(sportify_charts_2025, test_size=0.2, random_state=123)

# Prepare data for XGBoost
X_train = train_data.drop(columns='popularity')
y_train = train_data['popularity']
X_test = test_data.drop(columns='popularity')
y_test = test_data['popularity']

# Train the XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', eval_metric='rmse', max_depth=6, eta=0.1,
                         subsample=0.8, colsample_bytree=0.8, n_estimators=100, random_state=123)
xgb_model.fit(X_train, y_train)

# Make predictions
xgb_pred = xgb_model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, xgb_pred)

# Evaluate model performance
mae = mean_absolute_error(y_test, xgb_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, xgb_pred)

print("Mean Absolute Error (MAE):", mae)
print('mean squared error:', mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r_squared)

# Visualization: Actual vs Predicted
plot_data = pd.DataFrame({'Actual': y_test, 'Predicted': xgb_pred})
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_data, x='Actual', y='Predicted', alpha=0.6, color='blue')
plt.plot([plot_data.min().min(), plot_data.max().max()], [plot_data.min().min(), plot_data.max().max()],
         color='red', linestyle='dashed', linewidth=1)
plt.title("Actual vs Predicted Popularity (XGBoost)")
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.grid()
plt.show()

### testing on a new song
