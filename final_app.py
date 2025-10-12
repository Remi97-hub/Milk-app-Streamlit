import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="🐮Milk Quality Prediction App",layout="wide")

def load_file(filename):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        print("Warning : File Not Found")
        return None

st.title("🐮Milk Quality Prediction")
date=st.date_input("Date")


fat_model=load_file("Fat_prediction_Model_original_data.joblib")
fat_in_scaler=load_file("Fat_input_scaler_original_data.joblib")
fat_out_scaler=load_file("Fat_output_scaler_original_data.joblib")

snf_model=load_file("SNF_prediction_Model_original_data.joblib")
snf_in_scaler=load_file("Snf_input_scaler_original_data.joblib")
snf_out_scaler=load_file("Snf_output_scaler_original_data.joblib")

wa_model=load_file("Milk_Weighted_Average_Prediction.joblib")
wa_in_scaler=load_file("weighted_Average_input_scaler_original_data.joblib")
wa_out_scaler=load_file("weighted_Average_output_scaler_original_data.joblib")

st.header("Enter Quantities of the Societies")
quantities=[]
societies=["Eraiyur","Naripalayam","Ellaigramam","Kadiyar","Kolathur"]
cols=st.columns(5)



for i, s in enumerate(societies):
    with cols[i]:
        qty=st.number_input(f"{s}",min_value=0,key=f"q_{i}")
        quantities.append(qty)

total_qty=sum(quantities)
overall_fat=st.number_input("Enter Overall Fat")
overall_weighted_fat=(overall_fat*total_qty)/100

overall_snf=st.number_input("Enter Overall SNF")
overall_weighted_snf=(overall_snf*total_qty)/100

overall_wa=(overall_fat+overall_snf)*total_qty/100



#fat_in_sc=fat_in_scaler.transform(fat_in)
#snf_in_sc=snf_in_scaler.transform(snf_in)
#wa_in_sc=wa_in_scaler.transform(wa_in)

#result=[]
if st.button("Calculate"):
    fat_in=np.array([[overall_fat,overall_weighted_fat,*quantities]])
    snf_in=np.array([[overall_snf,overall_weighted_snf,*quantities]])
    wa_in=np.array([[overall_wa,*quantities]])
    fat_in_sc=fat_in_scaler.transform(fat_in)
    snf_in_sc=snf_in_scaler.transform(snf_in)
    wa_in_sc=wa_in_scaler.transform(wa_in)
    fat_pred=fat_model.predict(fat_in_sc)
    fat_pred_invsc=fat_out_scaler.inverse_transform(fat_pred.reshape(1,-1))
    snf_pred=snf_model.predict(snf_in_sc)
    snf_pred_invsc=snf_out_scaler.inverse_transform(snf_pred.reshape(1,-1))
    wa_pred=wa_model.predict(wa_in_sc)
    wa_pred_invsc=wa_out_scaler.inverse_transform(wa_pred.reshape(1,-1))
    #st.write(fat_pred_invsc)
    #st.write(snf_pred_invsc)
    #st.write(wa_pred_invsc)


    df=pd.DataFrame(index=societies,columns=["Quantity","Fat","SNF","Weighted Fat","Weighted SNF","Weighted Average"])
    #df["Society"]=societies
    df["Quantity"]=quantities
    df["Weighted Fat"]=fat_pred_invsc.flatten()
    df["Weighted SNF"]=snf_pred_invsc.flatten()
    df["Fat"]=np.round(df["Weighted Fat"]*100/df["Quantity"],1)
    df["SNF"]=np.round(df["Weighted SNF"]*100/df["Quantity"],1)
    df["Weighted Average"]=df["Weighted Fat"]+df["Weighted SNF"]

    total_qty=sum(quantities)
    total_fat=df["Fat"].sum()
    total_snf=df["SNF"].sum()
    total_wfa=df["Weighted Fat"].sum()
    total_wsnf=df["Weighted SNF"].sum()
    total_wa=df["Weighted Average"].sum()

    df.loc["Total"]=[total_qty,total_fat,total_snf,total_wfa,total_wsnf,total_wa]
    #multi_cols = pd.MultiIndex.from_product([[date], df.columns])
    date_level = [date] + [''] * (len(df.columns) - 1)
    df.columns = pd.MultiIndex.from_arrays([date_level, df.columns])
    st.dataframe(df)
    
    csv = df.reset_index().to_csv(index=False).encode("utf-8")
    
    st.download_button("⬇️ Download Results as CSV", csv, "milk_society_predictions.csv", "text/csv")
    

