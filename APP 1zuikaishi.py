import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

model_path = 'model_MLPllmin955551.pkl'
model = joblib.load(model_path)
import os


# Define feature names
feature_names = [
    "glucose", "albumin", "Scr", "Ka", "P", "NT.proBNP", "fibrinogen", "CRP"
]

# Streamlit 名字
st.title("Heart Disease Predictor")
model_path1 = 'min_max_scaler855551.pkl'
min_max_scaler = joblib.load(model_path1 )
# User inputs
import streamlit as st

glucose = st.number_input("glucose:", min_value=2.0, max_value=50.0, value=30.0)
albumin = st.number_input("albumin:", min_value=10.0, max_value=70.0, value=60.0)
Scr = st.number_input("Scr mg/dl (chol):", min_value=20.0, max_value=1000.0, value=100.0)
Ka = st.number_input("Ka:", min_value=0.1, max_value=10.0, value=1.0)
P = st.number_input("P:", min_value=0.1, max_value=20.0, value=5.0)
fibrinogen = st.number_input("fibrinogen:", min_value=0.5, max_value=15.0, value=1.0)
CRP = st.number_input("CRP:", min_value=0.0, max_value=15.0, value=1.0)
NT_proBNP = st.number_input("NT.proBNP:", min_value=0.1, max_value=142250.0, value=20.0)


# Process inputs and make predictions
feature_values = [glucose, albumin, Scr, Ka, P, fibrinogen, CRP, NT_proBNP]
features_scaled = np.array([feature_values])
# 使用加载的 MinMaxScaler 对用户输入的数据进行归一化
features = min_max_scaler.transform(features_scaled)
# Load background data for SHAP
background_data_path =  'train_sample_5055551.csv'
background_data = pd.read_csv(background_data_path)

# 确保选择特定的列进行缩放
feature_names = ["glucose", "albumin", "Scr", "Ka", "P", "NT.proBNP", "fibrinogen", "CRP"]
background_data_scaled = min_max_scaler.transform(background_data[feature_names].values)

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. 死亡"
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")
    explainer_shap = shap.KernelExplainer(model.predict_proba, background_data_scaled)
    shap_values = explainer_shap(pd.DataFrame(features, columns=feature_names))

    # 获取 expected_value 和 shap_values 的正确部分
    if isinstance(explainer_shap.expected_value, list) or isinstance(explainer_shap.expected_value, np.ndarray):
        expected_value = explainer_shap.expected_value[1] if predicted_class == 1 else explainer_shap.expected_value[0]
        shap_values_for_class = shap_values.values[0, :, 1] if predicted_class == 1 else shap_values.values[0, :, 0]
    else:
        expected_value = explainer_shap.expected_value
        shap_values_for_class = shap_values.values[0, :]

    shap.force_plot(expected_value, shap_values_for_class, pd.DataFrame([feature_values], columns=feature_names),
                    matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

