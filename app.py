# Bibliotecas para importar
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import joblib
from joblib import load
from utils import MinMax  
import io

# Título da aplicação
st.write("# Sistema preditivo de evasão 🔮")

# Upload excel 
uploaded_file = st.file_uploader("Carregue um arquivo Excel contendo a base de dados", type=["xlsx"])

# Pipeline para preprocessar os dados
def pipeline(df):
    """
     Função para normalizar os dados.

     Argumentos:
         df: Insira um dataframe.

     Retorna:
         dataframe normalizado.
    """
    pipeline = Pipeline([
        ('min_max_scaler', MinMax())
    ])
    df_pipeline = pipeline.fit_transform(df)
    return df_pipeline

# Function to mapping target values
def mapear_valor(valor):
    """
     Função para mapear valores alvo em dataframe e criar uma representação de rótulo.

     Argumentos:
         df: Insira um valor.

     Retorna:
         rótulo transformado.
    """
    if valor == 0:
        return 'Propenso a evadir'
    elif valor == 1:
        return 'Propenso a se formar'
    else:
        return 'Desconhecido'  

# Predictions
if uploaded_file is not None:
    data_original = pd.read_excel(uploaded_file)
    data_normalized = pipeline(data_original)
    model = joblib.load('modelo_xgb.joblib')
    final_pred = model.predict(data_normalized)
    data_original['Predição'] = [mapear_valor(valor) for valor in final_pred]    

    # Show data processed
    st.write("Dados Processados:")
    st.write(data_original)

    # Create a botton to download data
    output = io.StringIO()
    data_original.to_csv(output, index=False)
    csv_data = output.getvalue()
    st.download_button(label="Baixar CSV", data=csv_data, file_name="dados_predição.csv")
    st.balloons()



