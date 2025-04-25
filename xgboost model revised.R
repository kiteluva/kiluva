### SPOTIFY DATA ANALYSIS ###
library(tidyverse)
library(lubridate)
library(caret)
library(corrplot)
library(ggplot2)
library(ranger)
library(xgboost)

# Load the dataset
file_path <- "C:/Users/HP ELITEBOOK 820 G4/OneDrive/Documents/DATA ANALYSIS & VISUALIZATION/Python/spotifyProject/universal_top_spotify_songs.new.csv"
if (!file.exists(file_path)) {
  stop("File not found. Please check the file path.")
}
sportify_charts_2025 <- read_csv(file_path)
print(head(sportify_charts_2025))
# b4 cleaning distribution
ggplot(data=sportify_charts_2025)+
  geom_bar(mapping=aes(x=popularity))+
  ggtitle('popularity dist. b4 cleaning')
# Add a date difference column
sportify_charts_2025 <- sportify_charts_2025 %>%
  mutate(days_out = as.numeric(difftime(ymd(snapshot_date), ymd(album_release_date), units = "days")))

# Remove duplicate rows
sportify_charts_2025 <- sportify_charts_2025 %>% distinct(spotify_id, .keep_all = TRUE)

# Remove null and unnecessary columns
columns_to_remove <- c("country", "snapshot_date", "name", "artists", "album_name", "album_release_date", "spotify_id")
sportify_charts_2025 <- sportify_charts_2025 %>% select(-all_of(columns_to_remove))

# Convert boolean columns to integers
sportify_charts_2025 <- sportify_charts_2025 %>% mutate(is_explicit = as.integer(is_explicit))

# Change duration from milliseconds to minutes
sportify_charts_2025 <- sportify_charts_2025 %>% mutate(duration_min = duration_ms / 60000) %>% select(-duration_ms)

# Handle missing values by replacing them with column means
sportify_charts_2025 <- sportify_charts_2025 %>% mutate(across(everything(), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Reorder and clean up unnecessary columns
columns_order <- c("popularity", "days_out", "duration_min", "daily_rank", 
                   "daily_movement", "weekly_movement", "is_explicit", "mode", 
                   "time_signature", "key", "danceability", "energy", "loudness", 
                   "liveness", "tempo", "instrumentalness", "acousticness", 
                   "speechiness", "valence")
sportify_charts_2025 <- sportify_charts_2025 %>%
  select(all_of(columns_order)) %>% arrange(desc(popularity))
print(head(sportify_charts_2025))
#check popularity distribution
ggplot(data=sportify_charts_2025)+
  geom_bar(mapping=aes(x=popularity))+
  ggtitle('popularity dist. after cleaning')
# Select the audio feature columns
audio_features <- sportify_charts_2025 %>% 
  select(popularity, daily_rank, daily_movement, weekly_movement, duration_min, 
         days_out, is_explicit, mode, danceability, energy, loudness, 
         speechiness, acousticness, instrumentalness, time_signature, 
         liveness, valence, key, tempo)

# Compute correlation matrix
correlation_matrix <- cor(audio_features)
print(correlation_matrix)

# Display correlation between popularity and other features
popularity_correlation <- correlation_matrix[, "popularity"]
print(popularity_correlation)

# Plot heatmap
corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

## XGBoost Model ##
# Train-test split
set.seed(123)
trainIndex <- createDataPartition(sportify_charts_2025$popularity, p = 0.8, list = FALSE)
train_data <- sportify_charts_2025[trainIndex, ]
test_data <- sportify_charts_2025[-trainIndex, ]

# Prepare X and y for training
X_train <- train_data %>% select(-popularity)
y_train <- train_data$popularity
X_test <- test_data %>% select(-popularity)
y_test <- test_data$popularity

train.dframe<- data.frame(X_train, y_train)
test.dframe<- data.frame(X_test, y_test)
# testing popularity didtribution
ggplot(data=train.dframe)+
  geom_bar(mapping=aes(x=y_train))+
  ggtitle('popularity dist. for training data')

ggplot(data=test.dframe)+
  geom_bar(mapping=aes(x=y_test))+
  ggtitle('popularity dist. for testing data')
# Train the XGBoost model
xgb_model <- xgboost(data = as.matrix(X_train), label = y_train, 
                     nrounds = 100, objective = "reg:squarederror", eta = 0.1, 
                     max_depth = 10, subsample = 0.9, colsample_bytree = 0.8, verbose = 0)

# Make predictions
xgb_pred <- predict(xgb_model, as.matrix(X_test))

# Evaluate model performance
mse <- mean((xgb_pred - y_test)^2)
mae <- mean(abs(xgb_pred - y_test))
rmse <- sqrt(mse)
r_squared <- cor(xgb_pred, y_test)^2

cat("Mean Absolute Error (MAE):", mae, "\n")
cat("Mean Squared Error (MSE):", mse, "\n")
cat("Root Mean Squared Error (RMSE):", rmse, "\n")
cat("R-squared (R²):", r_squared, "\n")

# Visualization: Actual vs Predicted
plot_data <- data.frame(Actual = y_test, Predicted = xgb_pred)
ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_smooth(method = 'lm', col='red')+
  labs(title = "Actual vs Predicted Popularity (XGBoost)",
       x = "Actual Popularity", y = "Predicted Popularity") +
  theme_minimal()

# Plotting residuals
residuals <- y_test - xgb_pred
ggplot(data.frame(Residuals = residuals), aes(x = Residuals)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.6) +
  labs(title = "Residuals Distribution", x = "Residuals (Actual - Predicted)", y = "Frequency") +
  theme_minimal()

# Regression line
lm_model <- lm(xgb_pred ~ y_test)
summary(lm_model)
cat("Slope of Regression Line:", coef(lm_model)[2], "\n")
cat("Intercept of Regression Line:", coef(lm_model)[1], "\n")
