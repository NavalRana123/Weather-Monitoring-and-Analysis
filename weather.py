import streamlit as st
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Weather Data Analysis", layout="wide")

def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

uploaded_file = st.file_uploader("📂 Upload your weather.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

    st.sidebar.header("🔎 Filter Data")
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    temp_min, temp_max = st.sidebar.slider("Temperature Range", float(df['temperature'].min()), float(df['temperature'].max()), (float(df['temperature'].min()), float(df['temperature'].max())))
    rainfall_min, rainfall_max = st.sidebar.slider("Rainfall Range", float(df['rainfall'].min()), float(df['rainfall'].max()), (float(df['rainfall'].min()), float(df['rainfall'].max())))

    filtered_df = df[(df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))]
    filtered_df = filtered_df[(filtered_df['temperature'] >= temp_min) & (filtered_df['temperature'] <= temp_max)]
    filtered_df = filtered_df[(filtered_df['rainfall'] >= rainfall_min) & (filtered_df['rainfall'] <= rainfall_max)]

    filtered_df['month'] = filtered_df['date'].dt.month

    st.subheader("📊 Filtered Dataset Preview")
    st.dataframe(filtered_df.head())

    st.sidebar.header("⬇️ Export Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download Filtered Data as CSV", data=csv, file_name="filtered_weather.csv", mime="text/csv")

    st.subheader("📉 Temperature vs Rainfall Scatter Plot")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(filtered_df['temperature'], filtered_df['rainfall'], alpha=0.6, color='purple')
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Rainfall (mm)")
    ax.set_title("Temperature vs Rainfall")
    st.pyplot(fig)

    st.subheader("📦 Monthly Temperature Boxplot")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(x=filtered_df['month'], y=filtered_df['temperature'], ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Monthly Temperature Distribution")
    st.pyplot(fig)

    st.subheader("📊 Statistical Summary (Filtered Data)")
    st.write(filtered_df.describe())
    st.write("**Mean Temperature:**", filtered_df['temperature'].mean())
    st.write("**Median Temperature:**", filtered_df['temperature'].median())
    st.write("**Mean Rainfall:**", filtered_df['rainfall'].mean())
    st.write("**Median Rainfall:**", filtered_df['rainfall'].median())
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("ℹ️ Dataset Info")
    st.write(df.describe())

    st.subheader("🧹 Missing Values")
    st.write(df.isnull().sum())

    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df.fillna(method='ffill', inplace=True)

    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day_name()
   
    le = LabelEncoder()
    df['Day_encoded'] = le.fit_transform(df['day'])   

    st.subheader("🌡 Temperature Trend Over Time")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['date'], df['temperature'], color='orange')
    ax.set_title("Daily Temperature Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    st.pyplot(fig)

    st.subheader("📅 Average Monthly Temperature")
    monthly_avg = df.groupby('month')['temperature'].mean()
    fig, ax = plt.subplots(figsize=(10,5))
    monthly_avg.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Average Monthly Temperature")
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature (°C)")
    st.pyplot(fig)

    st.subheader("🌧 Rainfall Distribution")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.histplot(df['rainfall'], bins=30, kde=True, color='blue', ax=ax)
    ax.set_title("Rainfall Distribution")
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("📈 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,6))
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Weather Data Correlation")
    st.pyplot(fig)

    numeric_df = df.select_dtypes(include=['number'])

    plt.figure(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.show()

    st.subheader("🔍 Insights")
    st.write("**Hottest Day:**")
    st.write(df.loc[df['temperature'].idxmax()])

    st.write("**Coldest Day:**")
    st.write(df.loc[df['temperature'].idxmin()])

    st.write("**Day with Most Rainfall:**")
    st.write(df.loc[df['rainfall'].idxmax()])
    st.write("**Day with least Rainfall:**")
    st.write(df.loc[df['rainfall'].idxmin()])