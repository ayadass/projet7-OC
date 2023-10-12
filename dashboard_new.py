import streamlit as st
import pandas as pd
import requests
import pickle


def request_prediction(data):
    headers = {"Content-Type": "application/json"}
    model_uri = "http://127.0.0.1:5000/invocations"

    response = requests.request(
        method="POST", headers=headers, url=model_uri, json=data
    )

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(
                response.status_code, response.text
            )
        )

    return response.json()


def display_shap_values(shap_df, client_id):
    client_shap_values = shap_df[shap_df["SK_ID_CURR"] == client_id].drop(columns=["SK_ID_CURR"])
    sorted_shap_values = client_shap_values.iloc[0].sort_values(key=abs, ascending=False)
    return sorted_shap_values.head(10)

def main():
    st.title("Credit Default Prediction")

    # Load data
    df = pd.read_csv('application_test_new.csv')
    shap_df = pd.read_csv('shap_values_new.csv', index_col=0)


    # Client ID selection
    client_id = st.selectbox("Select Client ID", df['SK_ID_CURR'].unique())

    # Predict button
    predict_btn = st.button("Predict")
    if predict_btn:

        # Get data for selected ID
        client_data = df[df['SK_ID_CURR'] == client_id].drop(columns=['SK_ID_CURR'])  # Drop 'SK_ID_CURR'
        client_data = client_data.fillna(0.0)

        # Convert all columns to appropriate types and create data dictionary
        instance_data = {}
        for column in client_data.columns:
            if pd.api.types.is_numeric_dtype(client_data[column]):
                instance_data[column] = float(client_data[column].values[0])
            else:
                instance_data[column] = client_data[column].values[0]

        data = {"instances": [instance_data]}

        prediction = None
        try:
            prediction = request_prediction(data)
            if prediction['predictions'][0] == 0:
                st.write("Félicitations! Votre demande de crédit a été acceptée pour les raisons suivantes:")
            else:
                st.write("Votre demande de crédit a été refusée en raison des raisons suivantes:")

        except Exception as e:
            st.error(str(e))

        # Load SHAP values and display influential features
        
        influential_features = display_shap_values(shap_df, client_id)
        st.write(influential_features)


if __name__ == "__main__":
    main()



