# Note: Replace **<YOUR_APPLICATION_TOKEN>** with your actual Application token

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

URL = "https://api.langflow.astra.datastax.com"
LANGFLOW_ID = os.getenv("LANGFLOW_ID")
FLOW_ID = os.getenv("FLOW_ID")
APPLICATION_TOKEN = os.getenv("APPLICATION_TOKEN")
ENDPOINT = "templatepdfrag" # You can set a specific endpoint name in the flow settings


def run_flow(message: str) -> dict:
    api_url = f"{URL}/lf/{LANGFLOW_ID}/api/v1/run/{ENDPOINT}"

    payload = {
        "input_value": message,
        "output_type": "chat",
        "input_type": "chat",
    }
    headers = {"Authorization": "Bearer " + APPLICATION_TOKEN, "Content-Type": "application/json"}
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

# Streamlit GUI
def main():
    st.title("Attention is all you need QnA")

    message = st.text_area("Message", placeholder="Ask question about the article...")

    if st.button("Run Flow"):
        if not message.strip():
            st.error("Please enter a proper question")
            return
        
        try:
            with st.spinner("Running query"):
                respone = run_flow(message)
            respone = respone["outputs"][0]["outputs"][0]["results"]["message"]["text"]
            st.markdown(respone)
        except Exception as e:
            st.error(str(e))
            

if __name__ == "__main__":
    main()