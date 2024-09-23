import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from numpy import mean, std
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Title and description
st.title("Insurance Claims Fraud Detection")
st.write("This app allows you to upload an insurance claims dataset, manually input data, and evaluate different machine learning classifiers.")

# Specify directory containing CSV files
directory = os.getcwd()

# List available CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Dropdown to select a file from the directory
file_name = st.selectbox("Select a CSV file", csv_files)

if file_name:
    # Load the CSV file into a pandas DataFrame
    file_path = os.path.join(directory, file_name)
    df = pd.read_csv(file_path)
    
    # Display basic data information
    st.subheader("Data Preview")
    st.write(df.head())
    
    # Preprocess the data
    df.drop(columns=['_c39'], inplace=True, errors='ignore')
    df.replace('?', np.nan, inplace=True)
    df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])
    df['property_damage'] = df['property_damage'].fillna(df['property_damage'].mode()[0])
    df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
    df['authorities_contacted'] = df['authorities_contacted'].fillna(df['authorities_contacted'].mode()[0])

    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    
    # Split the data
    y = df['fraud_reported']
    feat = list(df.columns)
    feat.remove('fraud_reported')
    X = df[feat]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
    
    # Handle class imbalance using SMOTE
    smt = SMOTE()
    X_train, y_train = smt.fit_resample(X_train, y_train)
    
    # Define models
    def get_stacking():
        level0 = list()
        level0.append(('DT', DecisionTreeClassifier(max_depth=10, random_state=5)))
        level0.append(('RF', RandomForestClassifier(n_estimators=500)))
        level0.append(('KNN', KNeighborsClassifier(5)))
        level0.append(('XGB', XGBClassifier(objective='binary:logistic', use_label_encoder=False)))
        level1 = LogisticRegression()
        model = StackingClassifier(estimators=level0, final_estimator=level1, cv=10)
        return model

    def get_models():
        models = dict()
        models['Decision Tree'] = DecisionTreeClassifier(max_depth=10)
        models['Random Forest'] = RandomForestClassifier(n_estimators=500)
        models['KNN'] = KNeighborsClassifier(5)
        models['XGBoost'] = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        models['Stacking'] = get_stacking()
        return models

    models = get_models()

    # Allow user to select models
    st.subheader("Select Classifiers")
    model_options = st.multiselect("Choose classifiers", ['Decision Tree', 'Random Forest', 'KNN', 'XGBoost', 'Stacking'])




# Initialize the models based on user selection
model_options = st.multiselect("Select Models", options=list(models.keys()), default=list(models.keys()))

if not model_options:
    st.warning("Please select at least one model to proceed.")
else:
    selected_models = {name: models[name] for name in model_options}

    # Manual data input form
    st.subheader("Manual Data Entry for Prediction")

# Create a wide layout for input fields with 4 rows
num_columns = 4  # Set to 4 for a 4x4 layout
columns = st.columns(num_columns)

input_data = {}
for idx, col in enumerate(feat):
    # Determine which column to use
    with columns[idx % num_columns]:  # Cycle through columns
        if df[col].dtype == 'object':
            input_data[col] = st.selectbox(f"Select {col}", df[col].unique())
        else:
            input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))

# Convert input_data to a DataFrame for prediction
input_df = pd.DataFrame([input_data])

# Preprocess the manually input data
for col in df.select_dtypes(include='object').columns:
    input_df[col] = le.transform(input_df[col])

if st.button("Predict Fraud for Manual Data"):
    st.subheader("Prediction Results")
    predictions = {}
    for name, model in selected_models.items():
        model.fit(X_train, y_train)  # Training the model
        pred = model.predict(input_df)
        predictions[name] = pred[0]
        st.write(f"{name}: {'Fraud' if pred[0] == 1 else 'Not Fraud'}")

    # Highlight best and worst models based on accuracy (using cross-validation)
    st.subheader("Model Evaluation Results")
    results, names = [], []

    if selected_models:  # Check if any models were selected
        for name, model in selected_models.items():
            scores = cross_val_score(model, X_train, y_train, scoring='accuracy', 
                                     cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=5), n_jobs=-1)
            results.append(scores)  # Store individual scores
            names.append(name)

        if results:  # Check if results is not empty
            mean_scores = [mean(score) for score in results]  # Calculate mean scores for highlighting
            best_model = names[np.argmax(mean_scores)]
            worst_model = names[np.argmin(mean_scores)]

            for i, name in enumerate(names):
                if name == best_model:
                    st.markdown(f"<span style='color:green;'>**Best Model**: {name} - Accuracy: {mean_scores[i]:.2f}</span>", unsafe_allow_html=True)
                elif name == worst_model:
                    st.markdown(f"<span style='color:red;'>**Worst Model**: {name} - Accuracy: {mean_scores[i]:.2f}</span>", unsafe_allow_html=True)
                else:
                    st.write(f"{name}: Accuracy: {mean_scores[i]:.2f}")

            # Optionally, plot the results using boxplot with custom colors
            fig, ax = plt.subplots()
            box = ax.boxplot(results, labels=names, showmeans=True, patch_artist=True)  # Use individual scores
            
            # Customize boxplot colors
            for i, box in enumerate(box['boxes']):
                if names[i] == "Stacking":  # Change color for the Stacking model
                    box.set_facecolor('lightblue')  # Change this to your desired color
                else:
                    box.set_facecolor('lightgray')  # Default color for other models

            ax.set_title("Algorithm Comparison")
            st.pyplot(fig)
        else:
            st.write("No models were evaluated.")
    else:
        st.write("No models were selected for evaluation.")


