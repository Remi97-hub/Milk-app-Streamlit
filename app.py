import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Milk Society Prediction App", layout="wide")

# ------------------------------
# Helper: Safe load function
# ------------------------------
def load_file(filename):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        st.warning(f"⚠️ Missing file: {filename}")
        return None

# ------------------------------
# Load models and scalers
# ------------------------------
fat_model = load_file("Fat_prediction_Model_original_data.joblib")
snf_model = load_file("SNF_prediction_Model_original_data.joblib")
wa_model  = load_file("Weighted_Average_prediction_Model_original_data.joblib")

fat_in_scaler  = load_file("Fat_input_scaler_original_data.joblib")
fat_out_scaler = load_file("Fat_output_scaler_original_data.joblib")
snf_in_scaler  = load_file("Snf_input_scaler_original_data.joblib")
snf_out_scaler = load_file("Snf_output_scaler_original_data.joblib")
wa_in_scaler   = load_file("weighted_Average_input_scaler_original_data.joblib")
wa_out_scaler  = load_file("weighted_Average_output_scaler_original_data.joblib")

# ------------------------------
# Prediction function (with scalers)
# ------------------------------
def predict(model, in_scaler, out_scaler, features):
    X = np.array(features).reshape(1, -1)
    if in_scaler: X = in_scaler.transform(X)
    y_pred = model.predict(X)
    if out_scaler: y_pred = out_scaler.inverse_transform(y_pred.reshape(-1, 1))
    return float(y_pred.flatten()[0])

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🥛 Milk Society Prediction App")
st.write("Predict FAT, SNF, and Weighted Average for 5 societies using overall values.")

societies = ["Eraiyur", "Naripalayam", "Ellaigramam", "Kadiyar", "Kolathur"]

st.header("Enter Quantity for Each Society")
cols = st.columns(5)
quantities = []
for i, s in enumerate(societies):
    with cols[i]:
        q = st.number_input(f"{s}", min_value=0.0, step=0.1, key=f"qty_{i}")
        quantities.append(q)

st.header("Enter Overall Weighted Values")
overall_weighted_fat = st.number_input("Overall Weighted FAT", min_value=0.0, step=0.1)
overall_weighted_snf = st.number_input("Overall Weighted SNF", min_value=0.0, step=0.1)
overall_weighted_avg = st.number_input("Overall Weighted Average", min_value=0.0, step=0.1)

if st.button("Predict"):
    results = []
    for qty, name in zip(quantities, societies):
        fat_pred = predict(fat_model, fat_in_scaler, fat_out_scaler, [qty, overall_weighted_fat])
        snf_pred = predict(snf_model, snf_in_scaler, snf_out_scaler, [qty, overall_weighted_snf])
        wa_pred  = predict(wa_model, wa_in_scaler, wa_out_scaler, [qty, overall_weighted_avg])

        results.append({
            "Society": name,
            "Quantity": qty,
            "Pred_Weighted_FAT": round(fat_pred, 2),
            "Pred_Weighted_SNF": round(snf_pred, 2),
            "Pred_Weighted_Avg": round(wa_pred, 2)
        })

    df = pd.DataFrame(results)

    # Summary rows
    totals = {
        "Society": "TOTAL",
        "Quantity": df["Quantity"].sum(),
        "Pred_Weighted_FAT": df["Pred_Weighted_FAT"].sum(),
        "Pred_Weighted_SNF": df["Pred_Weighted_SNF"].sum(),
        "Pred_Weighted_Avg": df["Pred_Weighted_Avg"].sum()
    }
    averages = {
        "Society": "AVERAGE",
        "Quantity": df["Quantity"].mean(),
        "Pred_Weighted_FAT": df["Pred_Weighted_FAT"].mean(),
        "Pred_Weighted_SNF": df["Pred_Weighted_SNF"].mean(),
        "Pred_Weighted_Avg": df["Pred_Weighted_Avg"].mean()
    }

    df = pd.concat([df, pd.DataFrame([totals, averages])], ignore_index=True)

    st.subheader("Prediction Results")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download as CSV", data=csv, file_name="milk_predictions.csv", mime="text/csv")
