import streamlit as st
import pandas as pd
import os 

# Import SQLAlchemy components
from sqlalchemy import create_engine, inspect, text 
from sqlalchemy.exc import OperationalError, ProgrammingError 
import mysql.connector 

# --- HARDCODED DATABASE CREDENTIALS (FOR TESTING ONLY) ---
HARDCODED_DB_HOST = "/cloudsql/crucial-citron-461315-t0:us-central1:streamlit-stosbx" # Your Cloud SQL Instance Connection Name (Unix Socket Path)
HARDCODED_DB_DATABASE = "test"
HARDCODED_DB_USER = "root"
HARDCODED_DB_PASSWORD = "TrumpMick2024!!" # Your actual DB password
HARDCODED_DB_PORT = "" 
# -------------------------------------------------------------------------------


# --- Database Connection and Caching ---

@st.cache_resource 
def get_db_engine():
    """Establishes and returns a SQLAlchemy Engine for MySQL using HARDCODED credentials."""
    try:
        # Construct connection string for MySQL via Unix socket (preferred for App Engine)
        # We need to tell SQLAlchemy/mysql.connector to use the Unix socket explicitly.
        connection_string = (
            f"mysql+mysqlconnector://{HARDCODED_DB_USER}:"
            f"{HARDCODED_DB_PASSWORD}@{HARDCODED_DB_DATABASE}" # Host is provided in connect_args
        )
        
        # THIS IS THE CRITICAL CHANGE: Use connect_args for unix_socket
        engine = create_engine(
            connection_string, 
            echo=False,
            connect_args={
                "unix_socket": HARDCODED_DB_HOST # Pass the socket path here
            }
        ) 
        
        # Test connection by executing a simple query
        with engine.connect() as connection:
            result = connection.execute(text("SELECT CURRENT_DATE()")).scalar()
        
        st.success(f"Successfully connected to the database! Current DB date: {result}")
        return engine
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        st.write("Please check:")
        st.write(f"- Hardcoded `HARDCODED_DB_HOST` (`{HARDCODED_DB_HOST}`) is correctly set as the Cloud SQL Instance Connection Name (Unix Socket Path).")
        st.write(f"- Hardcoded `HARDCODED_DB_USER` (`{HARDCODED_DB_USER}`) and `HARDCODED_DB_PASSWORD` are correct.")
        st.write("- Cloud SQL API is enabled for your project.")
        st.write("- App Engine service account (or the one impersonated via WIF) has `Cloud SQL Client` role on your project.")
        st.write(f"Full error details: {e}") # Provide more details if it's still failing
        st.stop() 

# --- Streamlit App UI ---
st.set_page_config(page_title="DB Connection Test")

st.title("Database Connection Test App")
st.write("Attempting to connect to the Google Cloud SQL (MySQL) database with hardcoded credentials via Unix socket...")

# Call the function to attempt database connection
engine = get_db_engine()

# If connection was successful, you can now try to query some data
if engine:
    st.write("Connection established. Attempting a simple query...")
    try:
        test_table_name = "ng032024" 
        df = pd.read_sql_query(f"SELECT COUNT(*) FROM `{test_table_name}`", engine)
        st.success(f"Successfully queried table `{test_table_name}`! Row count: {df.iloc[0,0]}")
    except Exception as e:
        st.error(f"Error executing test query: {e}")
        st.write(f"Please ensure table `{test_table_name}` exists in database `test` and the user `root` has SELECT privileges.")

st.info("If you see green boxes, the database connection is working.")