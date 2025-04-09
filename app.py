# kashti app
import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
datasets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Kasti App")
    st.text("Analyzing the Titanic Dataset")
    st.title("My App")

with datasets:
    st.title("Kashti Data")
    st.text("Loading the Titanic dataset")
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df.columns)
    st.subheader("Bar Chart of Sex")
    st.bar_chart(df['sex'].value_counts())
    st.subheader("Bar Chart of Class")
    st.bar_chart(df['class'].value_counts())
    st.bar_chart(df['age'].sample(10))

with features:
    st.title("Features")
    st.text("Feature selection and details")
    st.markdown("**Feature 1:** The chosen feature from the dataset")

with model_training:
    st.title("Model Training")
    st.text("Training the model with the dataset")
    
    input, display = st.columns(2)
    max_dept = input.slider("Select max depth of the trees", min_value=10, max_value=100, value=20, step=5)
    n_estimators = input.selectbox("Number of trees in the Random Forest", options=[50, 100, 200, 300])

    input_feature = input.text_input("Which feature should be used?", 'age')
    
    if input_feature in df.columns:
        # Machine learning model
        model = RandomForestRegressor(max_depth=max_dept, n_estimators=n_estimators)
        
        # Define x and y
        x = df[[input_feature]]
        y = df['fare']
        
        # Fit model
        model.fit(x, y)
        pred = model.predict(x)
        
        # Display metrics
        display.subheader("Mean Absolute Error")
        display.write(mean_absolute_error(y, pred))
        display.subheader("Mean Squared Error")
        display.write(mean_squared_error(y, pred))
        display.subheader("R2 Score")
        display.write(r2_score(y, pred))
    else:
        display.error("Feature not found in the dataset. Please enter a valid column name.")



    
    
    
