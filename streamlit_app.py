import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(page_title="Platform for Risk Prediction of Macrosomia",layout="wide",initial_sidebar_state='auto')

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('stacking_model.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    return model, scaler

model, scaler = load_models()

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
        fasting_glucose = st.number_input("Fasting Glucose (mmol/L)", 3.0, 10.0, 5.0)
        parity = st.selectbox("Pregnant Woman's Parity", [0, 1, 2, 3], index=0)
        
with col2:
        st.header("Maternal Metabolism")

        TPOAb = st.selectbox("Thyroid Peroxidase Antibodies", ["Negative", "Positive"], index=0)
        anti_tpo = st.selectbox("Anti-thyroid Peroxidase Antibodies", ["Negative", "Positive"], index=0)
        urine_ketone_bodies = st.selectbox("Max Intensity of Urine Ketone Bodies", ["Negative", "+", "++", "+++", "++++"], index=0)
        ft4 = st.number_input("Free FT4 (pmol/L)", 5.0, 20.0, 12.0)
        
    
with col3:
        st.header("Fetal Ultrasound")
        placental_thickness = st.number_input("Placental Thickness (mm)", 10.0, 50.0, 25.0)
        abdominal_circumference = st.number_input("Abdominal Circumference (mm)", 200.0, 400.0, 300.0)
        biparietal_diameter = st.number_input("Biparietal Diameter (mm)", 100.0, 500.0, 300.0)
        femur_length = st.number_input("Femur Length (mm)", 100.0, 500.0, 300.0)
        sd = st.number_input("Umbilical Artery S/D", 0.0, 8.0, 2.0)
        fetal_position = st.selectbox("Fetal Position", ["Cephalic", "Non-Cephalic"], index=0)
        gender = st.selectbox("Baby Gender", ["Male", "Female"], index=0)
        
    
gender_map = {"Male": 0, "Female": 1}
fetal_position_map = {"Cephalic": 0, "Non-Cephalic": 1}
tpoab_map = {"Negative": 0, "Positive": 1}
anti_tpo_map = {"Negative": 0, "Positive": 1}
urine_ketone_bodies_map = {"Negative": 0, "+": 1, "++": 2, "+++": 3, "++++": 4}

columns_to_normalize = ['病人年龄', 'BMI', '空腹葡萄糖', '10-20周游离FT4',
                        '25-32周婴儿双顶径', '25-32周婴儿头围', '25-32周婴儿腹围', 
                        '25-32周婴儿股骨长', '25-32周婴儿胎盘厚', '25-32周婴儿脐动脉S/D', '25-32周婴儿胎心', '1小时葡萄糖', '2小时葡萄糖']

data_to_normalize = {
        '病人年龄': 0,
        'BMI':bmi,
        "空腹葡萄糖": fasting_glucose,
        "10-20周游离FT4": ft4,
        "25-32周婴儿双顶径": biparietal_diameter,
        "25-32周婴儿头围": 0,
        "25-32周婴儿腹围": abdominal_circumference,
        "25-32周婴儿股骨长": femur_length,
        "25-32周婴儿胎盘厚": placental_thickness,
        '25-32周婴儿脐动脉S/D': sd,
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
        "1-32周尿酮体最大阳性强度": urine_ketone_bodies_map[urine_ketone_bodies]  # 映射尿糖
    }

data_to_normalize_df = pd.DataFrame([data_to_normalize])

scaled_features = scaler.transform(data_to_normalize_df)

input_df_1 = pd.DataFrame(scaled_features, columns=data_to_normalize_df.columns)[
        ["BMI", "空腹葡萄糖", "25-32周婴儿胎盘厚", "25-32周婴儿腹围", "10-20周游离FT4", "25-32周婴儿双顶径", "25-32周婴儿股骨长",'25-32周婴儿脐动脉S/D']
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
<div class="footer"><p><p>Developed by AIMSLab - Macrosomia Prediction System © 2025</p></div>''', unsafe_allow_html=True)
