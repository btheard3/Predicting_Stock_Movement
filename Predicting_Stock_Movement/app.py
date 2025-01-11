import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Title and Introduction
st.title("Predicting Stock Movement")
st.markdown("""
This app predicts the movement of a stock based on historical data. 
Enter a stock ticker and explore the analysis.
""")

# User Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

if start_date >= end_date:
    st.error("Start Date must be earlier than End Date.")
else:
    try:
        # Fetch Stock Data
        st.subheader(f"Stock Data for {ticker}")
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for {ticker}. Please check the ticker or date range.")
            st.stop()
        st.write("Data Preview:", data.tail())

        # Flatten column names if multi-level
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [' '.join(col).strip() for col in data.columns]


        # Identify Close and Volume columns dynamically
        close_column = next((col for col in data.columns if "Close" in col and ticker in col), None)
        volume_column = next((col for col in data.columns if "Volume" in col and ticker in col), None)

        if not close_column or not volume_column:
            st.error("Required columns (Close, Volume) not found in the data. Check ticker and data source.")
            st.stop()

        # Visualization: Stock Closing Prices
        st.subheader(f"Closing Price for {ticker} from {start_date} to {end_date}")
        fig = px.line(data, x=data.index, y=close_column, title=f"{ticker} Closing Price")
        st.plotly_chart(fig)

        # Feature Engineering: Add Moving Averages
        data["MA_10"] = data[close_column].rolling(window=10).mean()
        data["MA_50"] = data[close_column].rolling(window=50).mean()

        # Visualization: Closing Price with Moving Averages
        fig = px.line(data, x=data.index, y=[close_column, "MA_10", "MA_50"],
                      title=f"{ticker} Closing Price with Moving Averages")
        st.plotly_chart(fig)

        # Predicting Stock Movement
        st.subheader(f"Predicting Movement for {ticker}")
        data["Target"] = (data[close_column].shift(-1) > data[close_column]).astype(int)
        data = data.dropna()

        # Features and Target
        X = data[[close_column, volume_column, "MA_10", "MA_50"]]
        y = data["Target"]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Model Evaluation
        y_pred = model.predict(X_test)

        # Enhanced Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        fig = go.Figure(data=[go.Table(
            header=dict(values=["Metric"] + list(report_df.columns),
                        fill_color='blue',
                        align='left'),
            cells=dict(values=[report_df.index,  # Class labels
                               report_df['precision'].round(2), 
                               report_df['recall'].round(2), 
                               report_df['f1-score'].round(2), 
                               report_df['support'].astype(int)],  # Support as int
                       fill_color='purple',
                       align='left'))
        ])
        st.plotly_chart(fig)

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.5)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va="center", ha="center")
        st.pyplot(fig)

        # Feature Importance
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.write(importance)
        fig = px.bar(importance, x="Importance", y="Feature", orientation="h", title="Feature Importance")
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
