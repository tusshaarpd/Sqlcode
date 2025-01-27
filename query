import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.title("Advanced SQL Code Generator")
st.write("Provide a description in simple English, and this tool will generate the corresponding SQL query.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "mrm8488/t5-small-finetuned-wikiSQL"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def generate_sql_query(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# User input
user_input = st.text_area("Describe the SQL query you need:", placeholder="e.g., Show me the total sales by region for the last year.")

if st.button("Generate SQL Query"):
    if user_input.strip():
        with st.spinner("Generating SQL query..."):
            sql_query = generate_sql_query(user_input)
        st.subheader("Generated SQL Query:")
        st.code(sql_query, language="sql")
    else:
        st.warning("Please provide a description to generate the query.")
