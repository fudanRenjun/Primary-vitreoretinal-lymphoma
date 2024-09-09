import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载随机森林模型
model = joblib.load('RF-LBL.pkl')

# 定义特征名称（根据你的数据调整）
feature_names = [
    "PDW", "Monocyte%", "PLCR", "Monocyte", "HG", "Basophil"
]

# Streamlit 用户界面
st.title("Primary vitreoretinal lymphoma(PVRL) Prediction App")

# 用户输入特征数据
pdw = st.number_input("PDW:", min_value=0.0, max_value=100.0, value=8.9)
monocyte_percent = st.number_input("Monocyte%(%):", min_value=0.0, max_value=100.0, value=6)
plcr = st.number_input("PLCR(%):", min_value=0.0, max_value=100.0, value=17.4)
monocyte = st.number_input("Monocyte(10^9/L):", min_value=0.0, max_value=10.0, value=0.42)
hg = st.number_input("HG(g/L):", min_value=0.0, max_value=200.0, value=145.0)
basophil = st.number_input("Basophil(10^9/L):", min_value=0.0, max_value=1.0, value=0.03)

# 将输入的数据转化为模型的输入格式
feature_values = [
    pdw, monocyte_percent, plcr, monocyte, hg, basophil
]
features = np.array([feature_values])

# 当点击按钮时进行预测
if st.button("Predict"):
    # 进行预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (0: Healthy, 1: PVRL)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果提供建议
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of PVRL. "
            f"The model predicts that your probability of having PVRL is {probability:.1f}%. "
            "We recommend consulting with a healthcare provider for further tests and evaluation."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of PVRL. "
            f"The model predicts that your probability of not having PVRL is {probability:.1f}%. "
            "However, it's still important to maintain regular health check-ups."
        )

    st.write(advice)

    # 计算并显示SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 根据预测结果生成并显示SHAP force plot
    if predicted_class == 1:
        shap.force_plot(explainer.expected_value[1], shap_values[:, :, 1],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    else:
        shap.force_plot(explainer.expected_value[0], shap_values[:, :, 0],
                        pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    # 保存SHAP图并显示
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
