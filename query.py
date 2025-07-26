import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Configure OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Please set OPENAI_API_KEY in your .env file")
    st.stop()

# Initialize LLM
llm = OpenAI(model_name="gpt-4o-mini", openai_api_key=api_key, temperature=0)

# Extract database schema information
def get_db_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_info = []
    for table in tables:
        table_name = table[0]
        # Skip sqlite internal tables
        if table_name.startswith('sqlite_'):
            continue
            
        # Get table structure
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        column_info = []
        for col in columns:
            column_info.append(f"  - {col[1]} ({col[2]})")
        
        # Get sample data (first 3 rows)
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
        sample_rows = cursor.fetchall()
        sample_data = "\n  Sample data:\n    " + "\n    ".join(str(row) for row in sample_rows)
        
        schema_text = f"Table: {table_name}\nColumns:\n" + "\n".join(column_info) + "\n" + sample_data
        schema_info.append(schema_text)
    
    conn.close()
    return "\n\n".join(schema_info)

# Setup RAG pipeline
def setup_rag_pipeline(db_path):
    # Get database schema
    schema_info = get_db_schema(db_path)
    
    # Create context document with schema information and query examples
    context = f"""
Database Schema Information:
{schema_info}

Instructions for SQL Generation:
1. Analyze the question and identify the tables and columns needed
2. Write a valid SQLite SQL query to answer the question
3. Return ONLY the SQL query, without any explanations or markdown formatting
4. Use appropriate JOIN operations if needed - STUDENT, COURSES, and ENROLLMENTS tables can be linked by joining on the appropriate ID fields
5. Use appropriate aliases for table names to make the query more readable
6. Use aggregate functions (COUNT, SUM, AVG, etc.) when appropriate
7. Accommodate for variations in how non-technical users might phrase questions

Example Join Relationships:
- STUDENT.STUDENT_ID joins with ENROLLMENTS.STUDENT_ID
- COURSES.COURSE_ID joins with ENROLLMENTS.COURSE_ID

Example Queries and Variations:
Question: Which students are enrolled in Machine Learning?
Alternative phrasings: "Show students taking Machine Learning" or "Who is in the Machine Learning course?"
SQL: SELECT s.NAME FROM STUDENT s JOIN ENROLLMENTS e ON s.STUDENT_ID = e.STUDENT_ID JOIN COURSES c ON e.COURSE_ID = c.COURSE_ID WHERE c.COURSE_NAME = 'Machine Learning';

Question: What's the average grade of students taking courses taught by Dr. Smith?
Alternative phrasings: "Average performance in Dr. Smith's classes" or "How well do Dr. Smith's students do?"
SQL: SELECT AVG(s.MARKS) FROM STUDENT s JOIN ENROLLMENTS e ON s.STUDENT_ID = e.STUDENT_ID JOIN COURSES c ON e.COURSE_ID = c.COURSE_ID WHERE c.INSTRUCTOR = 'Dr. Smith';

Question: Which course has the most student enrollments?
Alternative phrasings: "Most popular course" or "Course with highest enrollment"
SQL: SELECT c.COURSE_NAME, COUNT(e.STUDENT_ID) AS enrollment_count FROM COURSES c JOIN ENROLLMENTS e ON c.COURSE_ID = e.COURSE_ID GROUP BY c.COURSE_NAME ORDER BY enrollment_count DESC LIMIT 1;

Question: Show me all students in section A who are enrolled in courses with more than 3 credits
Alternative phrasings: "Section A students in high-credit courses" or "Who from section A is taking courses worth more than 3 credits?"
SQL: SELECT DISTINCT s.NAME FROM STUDENT s JOIN ENROLLMENTS e ON s.STUDENT_ID = e.STUDENT_ID JOIN COURSES c ON e.COURSE_ID = c.COURSE_ID WHERE s.SECTION = 'A' AND c.CREDITS > 3;
"""
    
    # Create document for RAG
    docs = [Document(page_content=context)]
    
    # Create text splitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(docs)
    
    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    
    return qa

# Get SQL query using RAG
def get_sql_query(question, qa):
    # Enhance prompt to get cleaner SQL and handle edge cases better
    prompt = f"""
    Generate a SQL query to answer this question from a non-technical user: "{question}"
    
    Make sure your query:
    1. Is valid SQLite syntax
    2. Uses appropriate WHERE clauses, JOINs, and functions
    3. Returns exactly what the user is asking for
    4. Handles common ways users might phrase their questions
    
    Return ONLY the SQL query with no explanations or additional text.
    """
    
    response = qa.run(prompt)
    
    # Clean response to ensure it's just SQL
    response = response.strip()
    if response.lower().startswith("sql:"):
        response = response[4:].strip()
    
    # Remove any markdown formatting
    if response.startswith("```sql"):
        response = response.replace("```sql", "").replace("```", "").strip()
    elif response.startswith("```"):
        response = response.replace("```", "").strip()
        
    return response

# Execute SQL query on SQLite database
def execute_sql_query(sql, db_path):
    conn = sqlite3.connect(db_path)
    try:
        # Use pandas to get better formatted results
        df = pd.read_sql_query(sql, conn)
        conn.close()
        
        # Improve column readability by renaming aggregate functions
        rename_dict = {}
        for col in df.columns:
            if col.startswith('AVG('):
                rename_dict[col] = 'Average'
            elif col.startswith('COUNT('):
                rename_dict[col] = 'Count'
            elif col.startswith('SUM('):
                rename_dict[col] = 'Total'
            elif col.startswith('MAX('):
                rename_dict[col] = 'Maximum'
            elif col.startswith('MIN('):
                rename_dict[col] = 'Minimum'
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
            
        # Make column names more readable for non-technical users
        df.columns = [col.replace('_', ' ').title() for col in df.columns]
        
        return df
    except Exception as e:
        conn.close()
        error_msg = str(e)
        # Make error messages more user-friendly
        if "no such column" in error_msg.lower():
            return pd.DataFrame({'I Need More Information': ["I'm not sure what you're asking about. Please provide more details."]})
        elif "syntax error" in error_msg.lower():
            return pd.DataFrame({'I Need More Information': ["I didn't understand your question. Could you rephrase it?"]})
        else:
            return pd.DataFrame({'I Need More Information': ["I couldn't find an answer to that question. Could you try asking differently?"]})

# SQLite Database Setup
def setup_database(db_path="student.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create STUDENT table if it doesn't exist
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='STUDENT';")
    student_table_exists = cur.fetchone()
    
    if not student_table_exists:
        cur.execute("""CREATE TABLE STUDENT(
                    STUDENT_ID INTEGER PRIMARY KEY,
                    NAME TEXT, 
                    CLASS TEXT, 
                    SECTION TEXT, 
                    MARKS INTEGER);""")
        
        # Insert sample student data
        sample_data = [
            (1, "Krish", "Data Science", "A", 90),
            (2, "Sudhanshu", "Data Science", "B", 100),
            (3, "Darius", "Data Science", "A", 86),
            (4, "Vikash", "DEVOPS", "A", 50),
            (5, "Dipesh", "DEVOPS", "A", 35),
            (6, "Maria", "Web Development", "B", 95),
            (7, "Alex", "Web Development", "A", 88),
            (8, "Priya", "Data Science", "B", 92),
            (9, "James", "DEVOPS", "B", 75),
            (10, "Sophia", "Web Development", "A", 67)
        ]
        
        cur.executemany("INSERT INTO STUDENT VALUES (?, ?, ?, ?, ?);", sample_data)
    
    # Create COURSES table if it doesn't exist
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='COURSES';")
    courses_table_exists = cur.fetchone()
    
    if not courses_table_exists:
        cur.execute("""CREATE TABLE COURSES(
                    COURSE_ID INTEGER PRIMARY KEY,
                    COURSE_NAME TEXT,
                    INSTRUCTOR TEXT,
                    CREDITS INTEGER,
                    SEMESTER TEXT);""")
        
        # Insert sample course data
        courses_data = [
            (101, "Python Programming", "Dr. Smith", 3, "Fall"),
            (102, "Database Management", "Prof. Johnson", 4, "Spring"),
            (103, "Machine Learning", "Dr. Williams", 4, "Fall"),
            (104, "Cloud Computing", "Prof. Davis", 3, "Spring"),
            (105, "Web Technologies", "Dr. Miller", 3, "Fall"),
            (106, "Data Visualization", "Prof. Wilson", 2, "Spring"),
            (107, "DevOps Practices", "Dr. Brown", 3, "Fall"),
            (108, "AI Fundamentals", "Prof. Taylor", 4, "Spring")
        ]
        
        cur.executemany("INSERT INTO COURSES VALUES (?, ?, ?, ?, ?);", courses_data)
    
    # Create ENROLLMENTS table to link students with courses
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ENROLLMENTS';")
    enrollments_table_exists = cur.fetchone()
    
    if not enrollments_table_exists:
        cur.execute("""CREATE TABLE ENROLLMENTS(
                    ENROLLMENT_ID INTEGER PRIMARY KEY,
                    STUDENT_ID INTEGER,
                    COURSE_ID INTEGER,
                    ENROLLMENT_DATE TEXT,
                    GRADE TEXT,
                    FOREIGN KEY(STUDENT_ID) REFERENCES STUDENT(STUDENT_ID),
                    FOREIGN KEY(COURSE_ID) REFERENCES COURSES(COURSE_ID));""")
        
        # Insert sample enrollment data
        enrollments_data = [
            (1, 1, 101, "2023-09-01", "A"),
            (2, 1, 103, "2023-09-02", "A-"),
            (3, 2, 101, "2023-08-15", "A+"),
            (4, 2, 102, "2023-08-16", "B+"),
            (5, 3, 103, "2023-09-10", "B"),
            (6, 3, 108, "2023-09-11", "A"),
            (7, 4, 104, "2023-08-20", "C+"),
            (8, 4, 107, "2023-08-21", "B-"),
            (9, 5, 107, "2023-09-05", "C"),
            (10, 6, 105, "2023-08-25", "A"),
            (11, 7, 105, "2023-09-15", "A-"),
            (12, 8, 103, "2023-08-30", "A"),
            (13, 9, 104, "2023-09-20", "B"),
            (14, 10, 105, "2023-09-25", "B+"),
            (15, 1, 106, "2023-10-01", "B+"),
            (16, 2, 108, "2023-10-05", "A"),
            (17, 6, 102, "2023-10-10", "A-"),
            (18, 7, 106, "2023-10-15", "B"),
            (19, 8, 108, "2023-10-20", "A-"),
            (20, 9, 107, "2023-10-25", "B-")
        ]
        
        cur.executemany("INSERT INTO ENROLLMENTS VALUES (?, ?, ?, ?, ?);", enrollments_data)
    
    conn.commit()
    conn.close()
    return db_path

# Streamlit App
def main():
    st.set_page_config(page_title="AI Database Assistant", layout="wide")
    st.title("Ask Questions About Your Student Database in Plain English")
    st.markdown("Simply ask a question about students, courses, and enrollments - I'll find the answer for you!")
    
    # Setup database
    db_path = setup_database()
    
    # Add some example questions to help users get started
    st.markdown("### Examples of questions you can ask:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- Which students are enrolled in Machine Learning?")
        st.markdown("- Show me all courses taught by Dr. Smith")
        st.markdown("- What's the average mark of students taking AI Fundamentals?")
    with col2:
        st.markdown("- Which student has the most course enrollments?")
        st.markdown("- List all courses with more than 3 credits")
        st.markdown("- How many students got an A grade in Python Programming?")
    
    # Technical details hidden by default
    with st.expander("Technical Details (for administrators)", expanded=False):
        schema_info = get_db_schema(db_path)
        st.code(schema_info)
    
    # Make the input box more prominent
    st.markdown("### What would you like to know about students and courses?")
    question = st.text_area("Type your question here:", height=80, 
                           placeholder="Example: Which students are enrolled in courses taught by Dr. Smith?")
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        search_button = st.button("🔍 Find Answer", use_container_width=True, type="primary")
    
    if search_button:
        if not question:
            st.warning("Please enter a question about the student or course data.")
        else:
            # Create progress indication
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Understanding the question
            status_text.text("Understanding your question...")
            progress_bar.progress(25)
            
            # Initialize RAG pipeline
            qa_pipeline = setup_rag_pipeline(db_path)
            
            # Step 2: Formulating the answer
            status_text.text("Finding the answer...")
            progress_bar.progress(50)
            
            # Get SQL query
            sql_query = get_sql_query(question, qa_pipeline)
            progress_bar.progress(75)
            
            # Execute query
            results = execute_sql_query(sql_query, db_path)
            progress_bar.progress(100)
            status_text.empty()
            
            # Display results in a user-friendly way
            st.markdown("### Here's what I found:")
            
            # Show results
            if isinstance(results, pd.DataFrame):
                if len(results) > 0:
                    st.dataframe(results, use_container_width=True)
                    
                    # Add a count of results
                    st.caption(f"Found {len(results)} {'result' if len(results) == 1 else 'results'}")
                else:
                    st.info("No matching records found for your question.")
            else:
                st.error("Sorry, I couldn't find an answer to that question.")
            
            # Show the SQL query in an expandable section for those who are curious
            with st.expander("See how I found the answer (SQL Query)", expanded=False):
                st.code(sql_query, language="sql")

if __name__ == "__main__":
    main()
