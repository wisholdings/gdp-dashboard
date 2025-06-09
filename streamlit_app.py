import streamlit as st
import pandas as pd
import os # Still import os, but won't use for credentials here

# Import SQLAlchemy components
from sqlalchemy import create_engine, inspect, text 
from sqlalchemy.exc import OperationalError, ProgrammingError 
import mysql.connector 

# --- HARDCODED DATABASE CREDENTIALS (FOR TESTING ONLY) ---
# >>> REMOVE THESE FOR PRODUCTION OR USE ENVIRONMENT VARIABLES/SECRET MANAGER <<<
HARDCODED_DB_HOST = "/cloudsql/crucial-citron-461315-t0:us-central1:streamlit-stosbx" # Your Cloud SQL Instance Connection Name (Unix Socket Path)
HARDCODED_DB_DATABASE = "test"
HARDCODED_DB_USER = "root"
HARDCODED_DB_PASSWORD = "TrumpMick2024!!" # Your actual DB password
HARDCODED_DB_PORT = "" # Empty for Unix socket connection
# -------------------------------------------------------------------------------


# --- Database Connection and Caching ---

@st.cache_resource # Cache the database engine
def get_db_engine():
    """Establishes and returns a SQLAlchemy Engine for MySQL using HARDCODED credentials."""
    try:
        connection_string = (
            f"mysql+mysqlconnector://{HARDCODED_DB_USER}:"
            f"{HARDCODED_DB_PASSWORD}@{HARDCODED_DB_HOST}/{HARDCODED_DB_DATABASE}"
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
        st.write(f"- Cloud SQL instance connection name in `HARDCODED_DB_HOST` (`{HARDCODED_DB_HOST}` should be `/cloudsql/PROJECT_ID:REGION:INSTANCE_NAME`)")
        st.write("- Database user and password in `HARDCODED_DB_USER` and `HARDCODED_DB_PASSWORD`.")
        st.write("- Cloud SQL API is enabled for your project.")
        st.write("- App Engine service account (or the one impersonated via WIF) has `Cloud SQL Client` role on your project.")
        st.stop() 

# --- Streamlit App UI ---
st.set_page_config(page_title="DB Connection Test")

st.title("Database Connection Test App")
st.write("Attempting to connect to the Google Cloud SQL (MySQL) database with hardcoded credentials...")

# Call the function to attempt database connection
engine = get_db_engine()

# If connection was successful, you can now try to query some data
if engine:
    st.write("Connection established. Attempting a simple query...")
    try:
        # Example: Query a table you know exists, e.g., 'ng032024'
        # Replace 'ng032024' with an actual table name if it's different in your 'test' DB
        test_table_name = "ng032024" 
        df = pd.read_sql_query(f"SELECT COUNT(*) FROM `{test_table_name}`", engine)
        st.success(f"Successfully queried table `{test_table_name}`! Row count: {df.iloc[0,0]}")
    except Exception as e:
        st.error(f"Error executing test query: {e}")
        st.write(f"Please ensure table `{test_table_name}` exists in database `test` and the user `root` has SELECT privileges.")

st.info("If you see green boxes, the database connection is working.")