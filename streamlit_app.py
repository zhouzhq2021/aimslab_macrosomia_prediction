import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('stacking_model.pkl')
    scaler = joblib.load('z_score_scaler.pkl')
    return model, scaler

model, scaler = load_models()

st.set_page_config(page_title="Platform for Risk Prediction of Macrosomia",layout="wide",initial_sidebar_state='auto')

# Sidebar
with st.sidebar:

    st.markdown("""
        <style>
            /* 调整导航标题样式 */
            .stTitle {
                font-size: 1.8em !important;
                margin-bottom: 25px !important;
                color: #2c3e50 !important;
            }
            
            /* 单选框容器样式 */
            div[role="radiogroup"] {
                gap: 15px;
            }
            
            /* 单选框选项样式 */
            .stRadio > label {
                font-size: 1.1em !important;
                padding: 10px 15px !important;
                border-radius: 8px !important;
                transition: all 0.3s ease !important;
            }
            
            /* 鼠标悬停效果 */
            .stRadio > label:hover {
                background-color: #f5f6fa !important;
                transform: translateX(5px);
            }
            
            /* 选中状态样式 */
            .stRadio > label[data-baseweb="radio"]:has(input:checked) {
                background-color: #3498db !important;
                color: white !important;
                font-weight: 500 !important;
            }
        </style>
        """, unsafe_allow_html=True)

    st.title("Navigation")
    page = st.radio("Page Selection",  
                   ["Introduction", 
                    "User Guide",
                    "Prediction Platform"],
                   label_visibility="collapsed") 

      # 添加间隔装饰线
    st.markdown("---")
    
    # 添加辅助说明文字
    st.markdown("<div style='font-size:0.9em; color:#7f8c8d; margin-top:30px;'></div>", 
                unsafe_allow_html=True)

# Page routing
if page == "Introduction":
    st.title("Ensemble Learning-Based Risk Prediction of Macrosomia Occurrence Study")
    st.markdown("### About the Study")
    st.markdown(" Macrosomia, one of the most prevalent pregnancy complications, is clinically defined as neonatal birth weight more than 4000 grams irrespective of gestational age. This condition poses substantial health risks to both mothers and fetuses. The effective utilization of routine prenatal examination data for accurate macrosomia prediction holds critical significance in optimizing pregnancy outcomes and ensuring maternal-fetal health.")
    st.markdown(" \n This study aims to systematically integrate diverse prenatal parameters, including maternal physical records, biochemical test results, and fetal ultrasound. Through comprehensive data mining, we intend to identify potential risk factors for macrosomia and develop an advanced prediction model by combining multiple machine learning and deep learning approaches with ensemble learning methodology. The proposed model seeks to establish a more objective, comprehensive, and clinically applicable decision-support tool for macrosomia diagnosis.")
    
    st.markdown("### Stacking Model to Predict Macrosomia Occurrence")
    st.markdown("In this study, a Stacking Ensemble Model was designed to show superior performance in the task of macrosomia occurrence risk prediction. The 10-fold cross-validation results showed that the Accuracy of the model was **0.804**, the Recall was **0.813**, and the AUC was **0.891**. The following is the architecture diagram of the Stacking Ensemble Model, where four different base models were integrated in the base model layer, namely, CatBoost, RF, MLP, and KNN models. model, MLP model and KNN model are integrated, and each base model is independently parameterized by Bayesian optimization in the early stage to ensure that each base model performs optimally. The logistic regression model is chosen as the meta-model in meta layer, and four probabilistic predictions of base learners are combined to form the input features of the meta-model.")


    # img = Image.open('Stacking.png')  
    # st.image(img, caption='Stacking Ensemble Model Architecture', width=350)

    img = Image.open('Stacking.png')
    st.image(img, caption='Stacking Ensemble Model Architecture', width=370) 

elif page == "User Guide":

    st.title("User Guide")

    st.markdown("### Introduction to the 14 input model predictors")
    st.markdown("""
    | **Maternal Characteristics** | **Metabolic and Immunologic Indicators**       | **Fetal Ultrasound**       |
    |-------------------------------|----------------------------------------|----------------------------------------|
    | BMI                           | Thyroid Peroxidase Antibodies (1-20w) | Placenta Thickness (25-32w)       |
    | Pregnancy Week                | Anti-thyroid Peroxidase Antibodies(1-20w)  | Abdominal Circumference (25-32w)  |
    | Fasting Glucose               | Max Intensity of Urine Glucose (1-32w)  | Biparietal Diameter (25-32w)         |
    | Pregnant Woman's Parity       | Free FT4 (1-20w)                      | Femur Length (25-32w)                  |
    |                               |                                        | Fetal Position (25-32w)               |
    |                               |                                        | Baby Gender                           |
    """ )
    st.markdown(" \n The first eight characteristics in the table are core continuous predictors and the last six are core subtype predictors. The system consists of three dimensions: maternal characteristics, metabolic and immunologic indicators, and fetal ultrasound.")

    st.markdown("### Notes")
    st.markdown("""
    + Please follow the requirements in the indicator description to enter data on the platform that meets the clinical specification.
    + When entering indicator data, please note that the unit is consistent with the requirements of the platform.
    + When entering the indicator data, please note that the corresponding week of pregnancy is consistent with the platform.
    """)


elif page == "Prediction Platform":

    st.markdown('''
    <style>
        div.button-container {
            display: flex;
            justify-content: center;
            margin: 30px 0;
        }
    </style>''' , unsafe_allow_html=True)

    st.title("Interface of Risk Prediction for Macrosomia Occurrence")
    
    # Create input columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Maternal Characteristics")

        bmi = st.number_input("BMI", 15.0, 40.0, 25.0)
        gestational_weeks = st.number_input("Pregnancy Week", 20, 42, 30)
        fasting_glucose = st.number_input("Fasting Glucose (mmol/L)", 3.0, 10.0, 5.0)
        parity = st.selectbox("Pregnant Woman's Parity", [0, 1, 2, 3], index=0)
        
    with col2:
        st.header("Metabolic and Immunologic Indicators")

        TPOAb = st.selectbox("Thyroid Peroxidase Antibodies", ["Negative", "Positive"], index=0)
        anti_tpo = st.selectbox("Anti-thyroid Peroxidase Antibodies", ["Negative", "Positive"], index=0)
        urine_glucose = st.selectbox("Max Intensity of Urine Glucose", ["Negative", "+", "++", "+++", "++++"], index=0)
        ft4 = st.number_input("Free FT4 (pmol/L)", 5.0, 20.0, 12.0)
        
    
    with col3:
        st.header("Fetal Ultrasound")
        placental_thickness = st.number_input("Placental Thickness (mm)", 10.0, 50.0, 25.0)
        abdominal_circumference = st.number_input("Abdominal Circumference (mm)", 200.0, 400.0, 300.0)
        biparietal_diameter = st.number_input("Biparietal Diameter (mm)", 100.0, 500.0, 300.0)
        femur_length = st.number_input("Femur Length (mm)", 100.0, 500.0, 300.0)
        fetal_position = st.selectbox("Fetal Position", ["Cephalic", "Non-Cephalic"], index=0)
        gender = st.selectbox("Baby Gender", ["Male", "Female"], index=0)
        
    
    gender_map = {"Male": 0, "Female": 1}
    fetal_position_map = {"Cephalic": 0, "Non-Cephalic": 1}
    tpoab_map = {"Negative": 0, "Positive": 1}
    anti_tpo_map = {"Negative": 0, "Positive": 1}
    urine_glucose_map = {"Negative": 0, "+": 1, "++": 2, "+++": 3, "++++": 4}

    columns_to_normalize = ['病人年龄', '怀孕孕周', 'BMI', '空腹葡萄糖', '10-20周游离FT4',
                        '25-32周婴儿双顶径', '25-32周婴儿头围', '25-32周婴儿腹围', '25-32周婴儿股骨长', 
                        '25-32周婴儿胎盘厚', '25-32周婴儿脐动脉S/D', '25-32周婴儿胎心', '1小时葡萄糖', '2小时葡萄糖']

    data_to_normalize = {
        '病人年龄': 0,
        "怀孕孕周": gestational_weeks,
        'BMI':bmi,
        "空腹葡萄糖": fasting_glucose,
        "10-20周游离FT4": ft4,
        "25-32周婴儿双顶径": biparietal_diameter,
        "25-32周婴儿头围": 0,
        "25-32周婴儿腹围": abdominal_circumference,
        "25-32周婴儿股骨长": femur_length,
        "25-32周婴儿胎盘厚": placental_thickness,
        '25-32周婴儿脐动脉S/D': 0,
        '25-32周婴儿胎心': 0,
        '1小时葡萄糖': 0,
        '2小时葡萄糖': 0,
    }

    other_data = {
        "婴儿性别": gender_map[gender],  # 映射性别
        "孕妇产次": parity,
        "25-32周婴儿胎位": fetal_position_map[fetal_position],  # 映射胎位
        "1-20周甲状腺过氧化物酶抗体": tpoab_map[TPOAb],  # 映射TPOAb
        "1-20周抗甲状腺过氧化物酶抗体": anti_tpo_map[anti_tpo],  # 映射Anti-TPO
        "1-32周尿葡萄糖最大阳性强度": urine_glucose_map[urine_glucose]  # 映射尿糖
    }

    data_to_normalize_df = pd.DataFrame([data_to_normalize])

    scaled_features = scaler.transform(data_to_normalize_df)

    input_df_1 = pd.DataFrame(scaled_features, columns=data_to_normalize_df.columns)[
        ["BMI", "怀孕孕周", "空腹葡萄糖", "25-32周婴儿胎盘厚", "25-32周婴儿腹围", "10-20周游离FT4", "25-32周婴儿双顶径", "25-32周婴儿股骨长"]
    ]

    other_data_df = pd.DataFrame([other_data])

    input_df = pd.concat([input_df_1, other_data_df], axis=1)
    
    # Normalization and prediction
    if st.button("Predict Risk"):
            try:
                print(input_df)
                risk_prob = model.predict_proba(input_df)[0][1]
                    
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                    
                # Visual display
                col_result, col_gauge = st.columns(2)
                with col_result:
                    st.metric("Macrosomia Risk Probability", f"{risk_prob*100:.1f}%")
                        
                    with col_gauge:
                        gauge_html = f'''
                        <div style="width: 100%; background: #f0f2f6; border-radius: 10px; padding: 20px;">
                            <div style="width: {risk_prob*100}%; height: 20px; background: {'#ff4b4b' if risk_prob > 0.5 else '#4CAF50'}; 
                                border-radius: 5px; transition: 0.3s;"></div>
                            <p style="text-align: center; margin-top: 10px;">Risk Level Indicator</p>
                        </div>
                        '''
                        st.markdown(gauge_html, unsafe_allow_html=True)
                
                st.markdown(" ")
                    
                if risk_prob > 0.7:
                    st.error("🚨 High Risk: Recommend clinical consultation and further monitoring.")
                elif risk_prob > 0.4:
                    st.warning("⚠️ Moderate Risk: Suggest increased monitoring frequency, and consider additional clinical examinations.")
                else:
                    st.success("✅ Low Risk: Maintain routine prenatal care. Regular check-ups are recommended.")

            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")


# Add footer
st.markdown("---")

st.markdown('''
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
}
</style>
<div class="footer"><p>Developed by LZU - Zhongquan Zhou © 2025</p></div>''', unsafe_allow_html=True)

# <p>Developed by AIMSLab - Macrosomia Prediction System © 2025</p>
