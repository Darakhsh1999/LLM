import streamlit as st

def process_text(user_input):
    return f"LLM read in user input: {user_input}"


st.title("LLM Interaction GUI")
with st.form("text_input"):
    submitted = st.form_submit_button("Submit")

if submitted:
    user_input = st.text_area("Enter your text here", height=100)
    if not user_input:
        st.write("Please enter some text.")
    else:
        response = process_text(user_input)
        st.success("Generated Response:")
        st.write(response)