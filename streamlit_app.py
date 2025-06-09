import streamlit as st
import pandas as pd
import os 

# Import SQLAlchemy components
from sqlalchemy import create_engine, text 
import mysql.connector 

# --- HARDCODED DATABASE CREDENTIALS (FOR TESTING ONLY) ---
# >>> THESE CREDENTIALS AND THE IP WHITELISTING ARE SECURITY RISKS <<<
# >>> USE ENVIRONMENT VARIABLES/SECRET MANAGER FOR PRODUCTION <<<

# IMPORTANT: This must be your Cloud SQL Instance's PUBLIC IP Address.
# You provided this: 34.66.61.153
HARDCODED_DB_HOST = "34.66.61.153" 
HARDCODED_DB_DATABASE = "test"
HARDCODED_DB_USER = "root"
HARDCODED_DB_PASSWORD = "TrumpMick2024!!" # Your actual DB password
HARDCODED_DB_PORT = 3306 # Standard MySQL port
# -------------------------------------------------------------------------------


# --- Database Connection ---

@st.cache_resource # Cache the database engine
def get_db_engine():
    """Establishes and returns a SQLAlchemy Engine for MySQL using HARDCODED credentials and direct IP."""
    try:
        connection_string = (
            f"mysql+mysqlconnector://{HARDCODED_DB_USER}:"
            f"{HARDCODED_DB_PASSWORD}@{HARDCODED_DB_HOST}:"
            f"{HARDCODED_DB_PORT}/{HARDCODED_DB_DATABASE}"
        )
        
        engine = create_engine(connection_string, echo=False) 
        
        # Test connection by executing a simple query
        with engine.connect() as connection:
            result = connection.execute(text("SELECT CURRENT_DATE()")).scalar()
        
        st.success(f"Successfully connected to the database! Current DB date: {result}")
        return engine
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        st.write("Please check:")
        st.write(f"- Hardcoded `HARDCODED_DB_HOST` (`{HARDCODED_DB_HOST}`) is your Cloud SQL Public IP.")
        st.write(f"- Hardcoded `HARDCODED_DB_USER` (`{HARDCODED_DB_USER}`) and `HARDCODED_DB_PASSWORD` are correct.")
        st.write(f"- **CRITICAL:** The **public IP address of the machine running this Streamlit app** is whitelisted in your Cloud SQL instance's Authorized Networks (`0.0.0.0/0` or your specific IP/CIDR).")
        st.write(f"- Cloud SQL API is enabled for your project.")
        st.write(f"Full error details: {e}")
        st.stop() 

# --- Streamlit App UI ---
st.set_page_config(page_title="DB Connection Test")

st.title("Database Connection Test App (Direct IP)")
st.write("Attempting to connect to the Google Cloud SQL (MySQL) database with hardcoded public IP credentials...")

# Call the function to attempt database connection
engine = get_db_engine()

# If connection was successful, you can now try to query some data
if engine:
    st.write("Connection established. Attempting a simple query...")
    try:
        test_table_name = "ng032024" # Replace with an actual table name from your 'test' DB
        df = pd.read_sql_query(f"SELECT COUNT(*) FROM `{test_table_name}`", engine)
        st.success(f"Successfully queried table `{test_table_name}`! Row count: {df.iloc[0,0]}")
    except Exception as e:
        st.error(f"Error executing test query: {e}")
        st.write(f"Please ensure table `{test_table_name}` exists in database `test` and the user `root` has SELECT privileges.")

st.info("If you see green boxes, the database connection is working.")