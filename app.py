import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration & Titles
st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="centered")
st.title("🌾 Wheat Crop Yield Prediction in Kyrgyzstan")
st.write("Enter the soil characteristics below to estimate the potential crop yield.")

# 2. Safe Model Loading
@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_soil.model")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load the ML model. Error: {e}")
    st.stop()

# 3. User Input Interface
st.subheader("Soil Parameters")

# We display English names to the user, but map them to the exact Russian strings the model expects
region_mapping = {
    "Chuy Valley (Chernozems)": "Чуйская долина (черноземы)",
    "Issyk-Kul (Chestnut soils)": "Иссык-Куль (каштановые)",
    "Fergana Valley (Serozems)": "Ферганская долина (серые)",
    "Jalal-Abad (Brown mountain soils)": "Джалал-Абад (бурые горные)",
    "Osh Region (Meadow/Desert soils)": "Ошская обл. (луговые/пустынные)"
}

selected_region_en = st.selectbox("Region / Soil Type", list(region_mapping.keys()))
region_ru = region_mapping[selected_region_en] # Inner Russian string for the model

year = st.number_input("Prediction Year", min_value=2020, max_value=2030, value=2026, step=1)
gumus = st.slider("Humus Content (%) (0-30 cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
ph = st.slider("pH Level (Water)", min_value=4.0, max_value=9.0, value=7.0, step=0.1)
cec = st.slider("CEC (cmol(+)/kg)", min_value=5.0, max_value=50.0, value=25.0, step=1.0)
clay = st.slider("Texture (% Clay)", min_value=5.0, max_value=60.0, value=30.0, step=1.0)
stones = st.slider("Stoniness (%)", min_value=0.0, max_value=50.0, value=15.0, step=1.0)
uncertainty = st.slider("Uncertainty (±%)", min_value=0.0, max_value=50.0, value=15.0, step=1.0)

# 4. Prediction Logic
if st.button("Calculate Prediction", type="primary"):
    try:
        # Dictionary with original Russian features used during training
        input_data = {
            "Год": int(year),
            "%Гумус (0-30 см)": float(gumus),
            "pH (вода)": float(ph),
            "CEC (смоль(+)/кг)": float(cec),
            "Текстура (% глина)": float(clay),
            "%Каменистость": float(stones),
            "Неопределенность (±%)": float(uncertainty)
        }
        
        # Create a single-row DataFrame
        df_base = pd.DataFrame([input_data])
        
        # Manual One-Hot Encoding for regions (using the original Russian training feature format)
        regions_list = [
            "Чуйская долина (черноземы)",
            "Иссык-Куль (каштановые)",
            "Ферганская долина (серые)",
            "Джалал-Абад (бурые горные)",
            "Ошская обл. (луговые/пустынные)"
        ]
        
        for r in regions_list:
            column_name = f"Регион / Тип почвы_{r}"
            df_base[column_name] = 1.0 if r == region_ru else 0.0

        # Align with the exact feature order of the XGBoost model
        model_features = model.feature_names_in_
        df_final = pd.DataFrame(0.0, index=[0], columns=model_features)
        
        for col in model_features:
            if col in df_base.columns:
                df_final[col] = df_base[col].values
                
        # Run ML model prediction
        prediction = model.predict(df_final)[0]
        
        # Display Results in English
        st.markdown("---")
        st.success(f"📈 **Predicted Wheat Yield Potential:** {prediction:.2f} centners/ha")
        
    except Exception as e:
        st.error(f"An error occurred during calculation: {e}")