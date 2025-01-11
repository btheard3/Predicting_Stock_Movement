Predicting Stock Movement
This Streamlit app uses historical stock data to predict whether the closing price of a stock will go up or down the next day. It combines interactive visualizations with a machine learning model to provide insights into stock trends and movement.

Features
Interactive Inputs:

Select a stock ticker (e.g., AAPL, MSFT).
Choose a date range dynamically for fetching historical stock data.
Data Visualization:

Closing Price trends over time.

![image](https://github.com/user-attachments/assets/99edd0d8-0f5f-486b-8c8a-868cc45b2d93)


Moving averages (10-day and 50-day) for analyzing trends.

![image](https://github.com/user-attachments/assets/87b71265-fa56-4e76-986c-9e75d7fd2630)

Feature Engineering:

Moving Averages: Captures short-term and long-term trends.
Volume: The total shares traded daily.
Prediction Model:

Random Forest Classifier predicts the next day's stock movement:
1: Price goes up.
0: Price goes down.
Model Evaluation:

Classification Report with Precision, Recall, and F1-Score.
Confusion Matrix for visualizing true/false predictions.
Feature Importance: Understand the contribution of each feature to the prediction.
