import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model():
    model_name = "t5-small-finetuned-wikisql-sql-nl-nl-sql"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def generate_sql_query(input_text, tokenizer, model):
    try:
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
        query = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return query
    except Exception as e:
        return f"Error generating SQL query: {e}"

# Streamlit app layout
st.title("Advanced SQL Code Generator")
st.write("Enter your requirement in plain English, and the app will generate the SQL query for you.")

# Load model
tokenizer, model = load_model()

# User input
input_text = st.text_area("Enter your query requirement (e.g., 'Get all users who signed up in 2022'):")

if st.button("Generate SQL Query"):
    if input_text.strip():
        with st.spinner("Generating SQL query..."):
            query = generate_sql_query(input_text, tokenizer, model)
        st.subheader("Generated SQL Query:")
        st.code(query, language="sql")
    else:
        st.warning("Please enter a valid requirement.")

