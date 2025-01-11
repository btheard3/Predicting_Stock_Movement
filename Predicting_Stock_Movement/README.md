# Predicting Stock Movement

This Streamlit app uses historical stock data to predict whether the closing price of a stock will go up or down the next day. It combines interactive visualizations with a machine learning model to provide insights into stock trends and movement.

## Features

1. **Interactive Inputs**:

   - Select a stock ticker (e.g., AAPL, MSFT).
   - Choose a date range dynamically for fetching historical stock data.

2. **Data Visualization**:

   - Closing Price trends over time.
   - Moving averages (10-day and 50-day) for analyzing trends.

3. **Feature Engineering**:

   - Moving Averages: Captures short-term and long-term trends.
   - Volume: The total shares traded daily.

4. **Prediction Model**:

   - Random Forest Classifier predicts the next day's stock movement:
     - `1`: Price goes up.
     - `0`: Price goes down.

5. **Model Evaluation**:
   - Classification Report with Precision, Recall, and F1-Score.
   - Confusion Matrix for visualizing true/false predictions.
   - Feature Importance: Understand the contribution of each feature to the prediction.
