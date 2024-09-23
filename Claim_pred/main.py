import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from numpy import mean, std
import warnings

# Initialize the model dictionary
def get_models():
    models = {
        'DT': DecisionTreeClassifier(max_depth=10),
        'RF': RandomForestClassifier(n_estimators=500),
        'KNN': KNeighborsClassifier(5),
        'XGB': XGBClassifier(objective='binary:logistic', eval_metric='logloss'),
        'Stacking': StackingClassifier(estimators=[
            ('DT', DecisionTreeClassifier(max_depth=10)),
            ('RF', RandomForestClassifier(n_estimators=500)),
            ('KNN', KNeighborsClassifier(5))
        ], final_estimator=LogisticRegression())
    }
    return models

# Streamlit app layout
st.title("Insurance Fraud Detection")

# Dropdown for selecting the file
directory = os.getcwd()
file_names = [f for f in os.listdir(directory) if f.endswith('.csv')]
selected_file = st.selectbox("Select a CSV file", options=file_names)

if selected_file:
    # Load data from the selected CSV file
    df = pd.read_csv(selected_file)
    st.write("Data Preview:")
    st.write(df.head())  # Display the first few rows of the DataFrame

    # Preprocessing
    df.drop(columns=['_c39'], inplace=True, errors='ignore')
    df.replace('?', np.nan, inplace=True)
    df.fillna({
        'police_report_available': df['police_report_available'].mode()[0],
        'property_damage': df['property_damage'].mode()[0],
        'collision_type': df['collision_type'].mode()[0],
        'authorities_contacted': df['authorities_contacted'].mode()[0]
    }, inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    
    y = df.fraud_reported
    feat = list(df.columns)
    feat.remove('fraud_reported')
    x = df[feat]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance
    smt = SMOTE()
    X_train, y_train = smt.fit_resample(X_train, y_train)

    # Load models
    models = get_models()

    # Model selection
    model_options = st.multiselect("Select Models", options=list(models.keys()), default=list(models.keys()))

    if not model_options:
        st.warning("Please select at least one model to proceed.")
    else:
        selected_models = {name: models[name] for name in model_options}

        # Manual data input form
        st.subheader("Manual Data Entry for Prediction")

        # Create a wide layout for input fields with 4 rows
        num_columns = 4
        columns = st.columns(num_columns)

        input_data = {}
        for idx, col in enumerate(feat):
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

            # Model evaluation
            st.subheader("Model Evaluation Results")
            results, names = [], []

            for name, model in selected_models.items():
                scores = cross_val_score(model, X_train, y_train, scoring='accuracy', 
                                         cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=5), n_jobs=-1)
                results.append(scores)  # Store individual scores
                names.append(name)

            if results:
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

                # Plot results using boxplot with custom colors
                fig, ax = plt.subplots()
                box = ax.boxplot(results, labels=names, showmeans=True, patch_artist=True)

                # Customize boxplot colors
                for i, box in enumerate(box['boxes']):
                    if names[i] == "Stacking":
                        box.set_facecolor('lightblue')  # Change color for the Stacking model
                    else:
                        box.set_facecolor('lightgray')  # Default color for other models

                ax.set_title("Algorithm Comparison")
                st.pyplot(fig)
            else:
                st.write("No models were evaluated.")
else:
    st.info("Please select a CSV file to proceed.")
