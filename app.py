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

if st.button("Predict for All Societies"):

    results = []

    # Prepare base arrays
    qty_array = np.array(quantities)  # [Eraiyur, Naripalayam, Ellaigramam, Kadiyar, Kolathur]

    for i, society in enumerate(societies):

        # --- FAT MODEL ---
        fat_input = np.array([[overall_weighted_fat, overall_weighted_fat, *qty_array]])
        fat_input_scaled = fat_in_scaler.transform(fat_input)
        fat_pred_scaled = fat_model.predict(fat_input_scaled)
        fat_pred = fat_out_scaler.inverse_transform(fat_pred_scaled)[0][i]

        # --- SNF MODEL ---
        snf_input = np.array([[overall_weighted_snf, overall_weighted_snf, *qty_array]])
        snf_input_scaled = snf_in_scaler.transform(snf_input)
        snf_pred_scaled = snf_model.predict(snf_input_scaled)
        snf_pred = snf_out_scaler.inverse_transform(snf_pred_scaled)[0][i]

        # --- WEIGHTED AVERAGE MODEL ---
        wa_input = np.array([[overall_weighted_avg, *qty_array]])
        wa_input_scaled = wa_in_scaler.transform(wa_input)
        wa_pred_scaled = wa_model.predict(wa_input_scaled)
        wa_pred = wa_out_scaler.inverse_transform(wa_pred_scaled)[0][i]

        results.append({
            "Society": society,
            "Quantity": qty_array[i],
            "Pred_Weighted_FAT": round(fat_pred, 2),
            "Pred_Weighted_SNF": round(snf_pred, 2),
            "Pred_Weighted_Avg": round(wa_pred, 2)
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Add totals and averages
    total_qty = df["Quantity"].sum()
    avg_fat = df["Pred_Weighted_FAT"].mean()
    avg_snf = df["Pred_Weighted_SNF"].mean()
    avg_wa = df["Pred_Weighted_Avg"].mean()

    df.loc[len(df)] = ["TOTAL / AVERAGE", total_qty, avg_fat, avg_snf, avg_wa]

    # Show results
    st.subheader("Predicted Results for Each Society")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results as CSV", csv, "milk_society_predictions.csv", "text/csv")
