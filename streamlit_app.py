import streamlit as st
import pandas as pd
import os # Import os for environment variables

# Import SQLAlchemy components
from sqlalchemy import create_engine, inspect, text 
from sqlalchemy.exc import OperationalError, ProgrammingError 
import mysql.connector # Or use 'pymysql' if you prefer, ensure 'mysql-connector-python' is in requirements.txt

# --- Database Connection and Caching ---

@st.cache_resource # Cache the database engine
def get_db_engine():
    """Establishes and returns a SQLAlchemy Engine for MySQL using environment variables."""
    try:
        # Get DB credentials from environment variables
        DB_HOST = os.environ.get("DB_HOST")
        DB_DATABASE = os.environ.get("DB_DATABASE")
        DB_USER = os.environ.get("DB_USER")
        DB_PASSWORD = os.environ.get("DB_PASSWORD")
        DB_PORT = os.environ.get("DB_PORT", "") # Set to empty string for Unix socket connection

        # Basic validation for env vars
        if not all([DB_HOST, DB_DATABASE, DB_USER, DB_PASSWORD]):
            raise ValueError(f"One or more database environment variables are missing. "
                             f"DB_HOST: {DB_HOST}, DB_DATABASE: {DB_DATABASE}, DB_USER: {DB_USER}, DB_PASSWORD: {'***' if DB_PASSWORD else 'None'}")

        # Construct connection string for MySQL via Unix socket (preferred for App Engine)
        # DB_HOST will contain the socket path, so port is not used.
        connection_string = (
            f"mysql+mysqlconnector://{DB_USER}:"
            f"{DB_PASSWORD}@{DB_HOST}/{DB_DATABASE}"
        )
        
        # You might need to add a specific connect_args for unix_socket if direct path doesn't work.
        # But for App Engine's /cloudsql/ path, it usually works directly.
        
        engine = create_engine(connection_string, echo=False) 
        
        # Test connection by executing a simple query
        with engine.connect() as connection:
            result = connection.execute(text("SELECT CURRENT_DATE()")).scalar()
        
        st.success(f"Successfully connected to the database! Current DB date: {result}")
        return engine
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        st.write("Please check:")
        st.write(f"- Cloud SQL instance connection name in app.yaml (`{os.environ.get('DB_HOST')}` should be `/cloudsql/PROJECT_ID:REGION:INSTANCE_NAME`)")
        st.write("- Database user, password, and name in your GitHub Secrets/App Engine environment variables.")
        st.write("- Cloud SQL API is enabled.")
        st.write("- Service account used for App Engine has `Cloud SQL Client` role.")
        st.stop() # Stop the app if DB connection fails

# --- Streamlit App UI ---
st.set_page_config(page_title="DB Connection Test")

st.title("Database Connection Test App")
st.write("Attempting to connect to the Google Cloud SQL (MySQL) database...")

# Call the function to attempt database connection
engine = get_db_engine()

# If connection was successful, you can now try to query some data
if engine:
    st.write("Connection established. Attempting a simple query...")
    try:
        # Example: Query a table you know exists, e.g., 'ng032024' from your ingestion
        # Replace 'ng032024' with an actual table name if it's different in your 'test' DB
        test_table_name = "ng032024" 
        df = pd.read_sql_query(f"SELECT COUNT(*) FROM `{test_table_name}`", engine)
        st.success(f"Successfully queried table `{test_table_name}`! Row count: {df.iloc[0,0]}")
    except Exception as e:
        st.error(f"Error executing test query: {e}")
        st.write(f"Please ensure table `{test_table_name}` exists in database `test` and the user `root` has SELECT privileges.")

st.info("If you see green boxes, the database connection is working.")