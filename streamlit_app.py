import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import re
from dateutil.relativedelta import relativedelta
import calendar
import plotly.express as px
import plotly.graph_objects as go # Added for potential dual-axis or more complex plots
import numpy as np
import os 

# Import SQLAlchemy components
from sqlalchemy import create_engine, inspect, text 
from sqlalchemy.exc import OperationalError, ProgrammingError 
import mysql.connector 

# --- HARDCODED DATABASE CREDENTIALS (FOR TESTING ONLY) ---
# >>> REMOVE THESE FOR PRODUCTION OR USE ENVIRONMENT VARIABLES/SECRET MANAGER <<<
# >>> THESE CREDENTIALS AND THE IP WHITELISTING (0.0.0.0/0) ARE SECURITY RISKS IF LEFT PUBLIC <<<
HARDCODED_DB_HOST = "34.66.61.153" 
HARDCODED_DB_DATABASE = "test"
HARDCODED_DB_USER = "root"
HARDCODED_DB_PASSWORD = "TrumpMick2024!!" # Your actual DB password
HARDCODED_DB_PORT = 3306 
# -------------------------------------------------------------------------------

# Constants for contract generation
PRODUCT_SYMBOL = "NG"
FUTURES_MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}
FUTURES_MONTH_CODES_REV = {v: k for k, v in FUTURES_MONTH_CODES.items()}

# --- Database Connection and Caching ---

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
            connection.execute(text("SELECT 1")).scalar() # Just a simple test, no need for result
        
        return engine
    except Exception as e:
        st.error(f"Error connecting to database. Please ensure your Cloud SQL instance is running, "
                 f"its public IP is correct, and that the environment's outbound IP is authorized (`0.0.0.0/0` if on App Engine). "
                 f"Also check database user/password are correct. Full error: {e}")
        st.stop() 

@st.cache_data(ttl=3600) # Cache the list of generated symbols for 1 hour
def generate_forward_contract_symbols(start_dt: datetime, num_months_out: int):
    """
    Generates contract symbols for the next `num_months_out` months,
    starting from `start_dt`. Uses 2-digit year suffix.
    Returns a list of (symbol_name, contract_y, contract_m, expiration_datetime_obj).
    """
    generated_contracts = []
    current_processing_date = start_dt
    
    for _ in range(num_months_out):
        contract_y, contract_m = current_processing_date.year, current_processing_date.month
        year_suffix = f"{contract_y % 100:02d}"
        month_code = FUTURES_MONTH_CODES[contract_m]
        symbol_name = f"{PRODUCT_SYMBOL}{month_code}{year_suffix}"
        
        # Simplistic expiration, not strictly needed for this app, but kept for consistency
        exp_year_cand = contract_y
        exp_month_cand = contract_m - 1
        if exp_month_cand == 0: 
            exp_month_cand = 12
            exp_year_cand -= 1
        try:
            expiration_day = calendar.monthrange(exp_year_cand, exp_month_cand)[1]
            expiration_datetime_obj = datetime(exp_year_cand, exp_month_cand, expiration_day)
        except ValueError:
            expiration_datetime_obj = datetime(contract_y, contract_m, 1) 
        
        generated_contracts.append((symbol_name, contract_y, contract_m, expiration_datetime_obj))
        current_processing_date += relativedelta(months=1)
        
    return generated_contracts

@st.cache_data(ttl=3600)
def get_all_contract_table_names(product_symbol="ng"):
    """Fetches all relevant futures table names from the database."""
    engine = get_db_engine()
    inspector = inspect(engine)
    all_table_names = inspector.get_table_names()
    
    futures_tables = [
        t for t in all_table_names 
        if t.startswith(product_symbol.lower()) and re.match(rf"^{product_symbol.lower()}\d{{2}}\d{{4}}$", t)
    ]
    futures_tables.sort() 
    return futures_tables

@st.cache_data(ttl=3600)
def get_contract_data_from_db(table_name: str):
    """
    Fetches all available data for a given contract table, including OI and Settlement Price.
    Returns DataFrame with 'trade_date' as index, sorted ascending.
    """
    engine = get_db_engine()
    df = pd.DataFrame()
    try:
        # Fetch OI and Settlement Price
        query = f"SELECT trade_date, open_interest, settlement_price FROM `{table_name.lower()}` ORDER BY trade_date ASC"
            
        df = pd.read_sql_query(query, engine)
        
        if not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date # Convert to date objects
            df = df.set_index('trade_date') # Set trade_date as index
            
            # Ensure columns are numeric, coercing errors to NaN
            df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce')
            df['settlement_price'] = pd.to_numeric(df['settlement_price'], errors='coerce')
        
    except Exception as e:
        st.error(f"Error fetching data for table '{table_name}': {e}")
    return df

# --- Streamlit App UI ---

# --- 0. SIMPLE AUTHENTICATION STATE ---
# Authentication is skipped for direct DB connection test
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = True 

if not st.session_state['authenticated']:
    st.title("Login Required (Auth Disabled for Direct Test)")
    st.write("Authentication is temporarily disabled in this version for direct DB connection testing.")
    st.stop()
else:
    st.set_page_config(page_title="NG Futures Historical Data")
    st.title("Natural Gas Futures Historical Contract Data")
    st.write("View the historical Open Interest and Settlement Price for a selected Natural Gas futures contract.")

    st.info(f"Data in this app is refreshed from the database every hour using `@st.cache_data`. "
            f"Your ingestion script determines how frequently the database itself is updated.")

    st.sidebar.subheader("Futures Month Codes:")
    for month_num, code in FUTURES_MONTH_CODES.items():
        st.sidebar.write(f"**{code}**: {calendar.month_name[month_num]}")

    # --- Establish DB connection ---
    engine = get_db_engine()
    if not engine: # If get_db_engine stopped the app, this won't be reached
        st.stop()

    # --- Select Contract ---
    all_futures_table_names = get_all_contract_table_names(PRODUCT_SYMBOL)
    
    available_contract_symbols = [] 
    for table_name in all_futures_table_names:
        # Assuming table names are like ng032024
        match = re.match(rf"^{PRODUCT_SYMBOL.lower()}(\d{{2}})(\d{{4}})$", table_name)
        if match:
            month_num = int(match.group(1))
            year_full = int(match.group(2))
            month_code = FUTURES_MONTH_CODES.get(month_num)
            if month_code:
                # Format for display: NG + Month Code + 2-digit Year (e.g., NGU24 for NG Aug 2024)
                display_symbol = f"{PRODUCT_SYMBOL}{month_code}{year_full%100:02d}"
                available_contract_symbols.append(display_symbol)
    
    available_contract_symbols.sort() # Sort chronologically for easy selection

    if not available_contract_symbols:
        st.warning("No Natural Gas futures contracts found in the database. Please ensure your ingestion script has successfully populated data.")
        st.stop()
    
    # Determine default selection (e.g., the most recent contract or a common one)
    # Using the last item in the sorted list as a default
    default_selected_symbol = available_contract_symbols[-1] if available_contract_symbols else None

    selected_contract_symbol = st.selectbox(
        "Select a Futures Contract:", 
        options=available_contract_symbols,
        index=available_contract_symbols.index(default_selected_symbol) if default_selected_symbol else 0,
        help="Choose a Natural Gas futures contract to view its historical data."
    )

    if selected_contract_symbol:
        # Parse the selected symbol back to table name format (e.g., NGU24 -> ng082024)
        match = re.match(r"([A-Z]+)([FGHJKMNQUVXZ])(\d{1,2})", selected_contract_symbol)
        if match:
            product, month_code, year_suffix_str = match.groups()
            base_year_2digit = int(year_suffix_str)
            
            # Convert 2-digit year suffix to full year (robustly)
            current_century = (datetime.utcnow().year // 100) * 100
            if base_year_2digit <= (datetime.utcnow().year % 100) + 1: # Assuming years 00-current_year_last_2_digits + 1 are current century
                full_year = current_century + base_year_2digit
            else: # Assuming years > current_year_last_2_digits + 1 are previous century
                full_year = (current_century - 100) + base_year_2digit
            
            month_num = FUTURES_MONTH_CODES_REV[month_code]
            table_name_to_fetch = f"{PRODUCT_SYMBOL}{month_num:02d}{full_year}".lower()

            df_contract_data = get_contract_data_from_db(table_name_to_fetch)

            if not df_contract_data.empty:
                st.subheader(f"Historical Data for {selected_contract_symbol}")

                # Filter out NaN/zero Open Interest and Settlement Price for plotting
                df_plot = df_contract_data.copy()
                df_plot['open_interest'] = pd.to_numeric(df_plot['open_interest'], errors='coerce')
                df_plot['settlement_price'] = pd.to_numeric(df_plot['settlement_price'], errors='coerce')
                
                df_plot_filtered_oi = df_plot[df_plot['open_interest'].notna() & (df_plot['open_interest'] != 0)].copy()
                df_plot_filtered_settlement = df_plot[df_plot['settlement_price'].notna()].copy()


                # --- Plot 1: Historical Open Interest ---
                if not df_plot_filtered_oi.empty:
                    fig_oi = px.line(df_plot_filtered_oi, 
                                     x=df_plot_filtered_oi.index, 
                                     y="open_interest", 
                                     title=f"Historical Open Interest: {selected_contract_symbol}",
                                     labels={"x": "Trade Date", "open_interest": "Open Interest"},
                                     hover_data={"open_interest": ":,.0f", df_plot_filtered_oi.index.name: "|%Y-%m-%d"}) # Include index in hover
                    fig_oi.update_yaxes(rangemode="tozero")
                    st.plotly_chart(fig_oi, use_container_width=True)
                else:
                    st.info(f"No valid (non-zero) Open Interest data found for {selected_contract_symbol}.")

                # --- Plot 2: Historical Settlement Price ---
                if not df_plot_filtered_settlement.empty:
                    fig_settlement = px.line(df_plot_filtered_settlement, 
                                              x=df_plot_filtered_settlement.index, 
                                              y="settlement_price", 
                                              title=f"Historical Settlement Price: {selected_contract_symbol}",
                                              labels={"x": "Trade Date", "settlement_price": "Settlement Price"},
                                              hover_data={"settlement_price": ":,.2f", df_plot_filtered_settlement.index.name: "|%Y-%m-%d"})
                    fig_settlement.update_yaxes(rangemode="tozero")
                    st.plotly_chart(fig_settlement, use_container_width=True)
                else:
                    st.info(f"No valid Settlement Price data found for {selected_contract_symbol}.")

                st.subheader("Raw Data Sample:")
                st.dataframe(df_contract_data.head()) # Show a sample of the raw data
                st.markdown(f"Total rows: {len(df_contract_data)}")


            else:
                st.warning(f"No data found for contract {selected_contract_symbol} (table '{table_name_to_fetch}') in the database.")
        else:
            st.error("Could not parse selected contract symbol.")

    st.markdown("---") 
    st.write("Data sourced from Databento via your ingestion script.")