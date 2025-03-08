#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[11]:


# Load trained models
def load_models():
    models = {}
    try:
        models['Random Forest'] = pickle.load(open("rf.pkl", "rb"))
    except FileNotFoundError:
        st.warning("Random Forest model not found.")
    
    try:
        models['Decision Tree'] = pickle.load(open("clf.pkl", "rb"))
    except FileNotFoundError:
        st.warning("Decision Tree model not found.")
    
    try:
        models['SVM'] = pickle.load(open("model.pkl", "rb"))
    except FileNotFoundError:
        st.warning("SVM model not found.")
    
    try:
        models['Bagging'] = pickle.load(open("bag_model.pkl", "rb"))
    except FileNotFoundError:
        st.warning("Bagging model not found.")
    
    try:
        models['AdaBoost'] = pickle.load(open("adb.pkl", "rb"))
    except FileNotFoundError:
        st.warning("AdaBoost model not found.")
    
    try:
        models['XGBoost'] = pickle.load(open("xgb_clf.pkl", "rb"))
    except FileNotFoundError:
        st.warning("XGBoost model not found.")
    
    try:
        models['Voting Classifier'] = pickle.load(open("voting_clf.pkl", "rb"))
    except FileNotFoundError:
        st.warning("Voting Classifier model not found.")
    
    return models if models else None


# In[12]:


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Visa.csv")

def main():
    st.title("Visa Approval Prediction App")
    st.write("Predict whether a visa application will be approved based on various factors.")
    
    models = load_models()
    data = load_data()
    
    if not models:
        st.error("No models found. Please ensure at least one model file exists.")
        return
    
    # Model selection dropdown
    selected_model = st.selectbox("Select a Model", list(models.keys()))
    model = models.get(selected_model)
    
    # User inputs
    education = st.selectbox("Education Level", data["education_of_employee"].unique())
    experience = st.selectbox("Job Experience", ["Y", "N"])
    job_training = st.selectbox("Requires Job Training", ["Y", "N"])
    employees = st.number_input("Number of Employees", min_value=1, step=1)
    year_estab = st.number_input("Year of Establishment", min_value=1800, max_value=2025, step=1)
    region = st.selectbox("Region of Employment", data["region_of_employment"].unique())
    wage = st.number_input("Prevailing Wage", min_value=0.0, step=100.0)
    wage_unit = st.selectbox("Unit of Wage", data["unit_of_wage"].unique())
    full_time = st.selectbox("Full-Time Position", ["Y", "N"])
    
    # Convert inputs into dataframe
    input_data = pd.DataFrame({
        "education_of_employee": [education],
        "has_job_experience": [experience],
        "requires_job_training": [job_training],
        "no_of_employees": [employees],
        "yr_of_estab": [year_estab],
        "region_of_employment": [region],
        "prevailing_wage": [wage],
        "unit_of_wage": [wage_unit],
        "full_time_position": [full_time]
    })
    
    if st.button("Predict"):
        if model:
            prediction = model.predict(input_data)[0]
            accuracy = model.score(data.drop("case_status", axis=1), data["case_status"]) * 100
            st.success(f"Visa Application Status: {'Certified' if prediction == 1 else 'Denied'}")
            st.write(f"Model Accuracy: {accuracy:.2f}%")
        else:
            st.error("Prediction not available due to missing model.")


# In[13]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




