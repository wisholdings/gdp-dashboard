import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import re
from dateutil.relativedelta import relativedelta
import calendar
import plotly.express as px
import numpy as np
import os # Still import os, but won't use for credentials here

# Import SQLAlchemy components
from sqlalchemy import create_engine, inspect, text 
from sqlalchemy.exc import OperationalError, ProgrammingError 
import mysql.connector 

# --- HARDCODED DATABASE CREDENTIALS (FOR TESTING ONLY) ---
# >>> REMOVE THESE FOR PRODUCTION OR USE ENVIRONMENT VARIABLES/SECRET MANAGER <<<
# >>> THESE CREDENTIALS AND THE IP WHITELISTING (0.0.0.0/0) ARE SECURITY RISKS IF LEFT PUBLIC <<<
# IMPORTANT: This must be your Cloud SQL Instance's PUBLIC IP Address.
HARDCODED_DB_HOST = "34.66.61.153" 
HARDCODED_DB_DATABASE = "test"
HARDCODED_DB_USER = "root"
HARDCODED_DB_PASSWORD = "TrumpMick2024!!" # Your actual DB password
HARDCODED_DB_PORT = 3306 # Standard MySQL port
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
            result = connection.execute(text("SELECT CURRENT_DATE()")).scalar()
        
        # st.success(f"Database connection established. Current DB date: {result}") # Optional success message
        return engine
    except Exception as e:
        st.error(f"Error connecting to database. Please ensure your Cloud SQL instance is running, "
                 f"its public IP is correct, and that the environment's outbound IP is authorized (`0.0.0.0/0` if on App Engine). "
                 f"Also check database user/password are correct. Full error: {e}")
        st.stop() 

def _get_last_day_of_month(year, month):
    """Helper to get the last day of a given month."""
    return calendar.monthrange(year, month)[1]

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

        exp_year_cand = contract_y
        exp_month_cand = contract_m - 1
        if exp_month_cand == 0: 
            exp_month_cand = 12
            exp_year_cand -= 1
        try:
            expiration_day = _get_last_day_of_month(exp_year_cand, exp_month_cand)
            expiration_datetime_obj = datetime(exp_year_cand, exp_month_cand, expiration_day)
        except ValueError:
            expiration_datetime_obj = datetime(contract_y, contract_m, 1) 
        
        generated_contracts.append((symbol_name, contract_y, contract_m, expiration_datetime_obj))
        current_processing_date += relativedelta(months=1)
        
    return generated_contracts

@st.cache_data(ttl=3600)
def get_next_active_month_symbol(product_symbol="ng"):
    """
    Determines the first "next active month" contract symbol.
    This is typically the first contract whose expiration date is in the future.
    """
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1) 

    temp_contracts_details = generate_forward_contract_symbols(datetime.utcnow(), 60) 
    
    found_next_active_contract = None
    for sym, y, m, exp_dt_obj in temp_contracts_details:
        if exp_dt_obj.date() > yesterday: 
            found_next_active_contract = (sym, y, m, exp_dt_obj)
            break
            
    if found_next_active_contract:
        return found_next_active_contract
    else:
        st.warning("Could not determine the next active futures contract based on expiration. Defaulting to the next calendar month.")
        current_month_dt = datetime.utcnow() + relativedelta(months=1)
        temp_contracts_details_fallback = generate_forward_contract_symbols(current_month_dt, 1)
        if temp_contracts_details_fallback:
            return temp_contracts_details_fallback[0]
        else: 
            return (f"{PRODUCT_SYMBOL}{FUTURES_MONTH_CODES[current_month_dt.month]}{current_month_dt.year%100:02d}", 
                    current_month_dt.year, current_month_dt.month, current_month_dt.date())

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
    Fetches all available data for a given contract table.
    Returns DataFrame with 'trade_date' as index, sorted ascending.
    """
    engine = get_db_engine()
    df = pd.DataFrame()
    try:
        query = f"SELECT trade_date, open_interest FROM `{table_name.lower()}` ORDER BY trade_date ASC"
            
        df = pd.read_sql_query(query, engine)
        
        if not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date 
            df = df.set_index('trade_date') 
        
    except Exception as e:
        st.error(f"Error fetching data for table '{table_name}': {e}")
    return df

# --- Streamlit App UI ---

# --- 0. SIMPLE AUTHENTICATION STATE ---
# Removed os.environ.get for APP_USERNAME/PASSWORD as they are not used/hardcoded in this script.
# If you want authentication, you'd hardcode it here too or use Streamlit's secrets.toml for simplicity.
# For now, removed to focus on DB connection first.
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = True # Set to True to skip login for this example

# Simplified/removed authentication for direct testing
if not st.session_state['authenticated']:
    st.title("Login Required (Auth Disabled for Direct Test)")
    st.write("Authentication is temporarily disabled in this version for direct DB connection testing.")
    st.stop()
else:
    st.sidebar.success(f"Logged in (Authentication temporarily skipped).")
    # No logout button if not actually logging in
    # if st.sidebar.button("Logout"):
    #     st.session_state['authenticated'] = False
    #     st.experimental_rerun()

    st.title("Natural Gas Futures Historical Open Interest")
    st.write("Explore the historical Open Interest for specific Natural Gas futures contract months.")

    st.info(f"Data in this app is refreshed from the database every hour using `@st.cache_data`. "
            f"Your ingestion script determines how frequently the database itself is updated.")

    st.sidebar.subheader("Futures Month Codes:")
    for month_num, code in FUTURES_MONTH_CODES.items():
        st.sidebar.write(f"**{code}**: {calendar.month_name[month_num]}")

    # --- SECTION: Historical Open Interest Comparison (Simplified) ---
    st.header("Historical Open Interest Comparison")
    st.write("Compare the Open Interest history for the same contract month across different years.")

    # Establish DB connection first
    engine = get_db_engine()
    if not engine: # If get_db_engine stopped the app, this won't be reached
        st.stop()

    all_futures_table_names = get_all_contract_table_names(PRODUCT_SYMBOL)
    
    available_contract_symbols_for_history = [] 
    for table_name in all_futures_table_names:
        match = re.match(rf"^{PRODUCT_SYMBOL.lower()}(\d{{2}})(\d{{4}})$", table_name)
        if match:
            month_num = int(match.group(1))
            year_full = int(match.group(2))
            month_code = FUTURES_MONTH_CODES.get(month_num)
            if month_code:
                display_symbol = f"{PRODUCT_SYMBOL}{month_code}{year_full%100:02d}"
                available_contract_symbols_for_history.append(display_symbol)
    
    available_contract_symbols_for_history.sort() 


    if not available_contract_symbols_for_history:
        st.warning("No Natural Gas futures contracts found in the database for historical comparison. Please ensure your ingestion script has successfully populated data (e.g., from ng_futures_contract_history.db).")
    else:
        # get_next_active_month_symbol needs engine or should not try to connect to DB.
        # It just uses datetime.utcnow(), so it's fine.
        next_active_contract_tuple_for_history = get_next_active_month_symbol(datetime.utcnow()) 
        
        default_index_history = 0
        if next_active_contract_tuple_for_history and next_active_contract_tuple_for_history[0] in available_contract_symbols_for_history:
            default_index_history = available_contract_symbols_for_history.index(next_active_contract_tuple_for_history[0])

        selected_reference_contract_symbol = st.selectbox(
            "Select a Reference Contract Month (e.g., NGH25):", 
            options=available_contract_symbols_for_history,
            index=default_index_history,
            help="Choose a futures contract month (e.g., NGH25 for March 2025) to compare its Open Interest history across multiple years."
        )

        num_years_to_compare = st.slider(
            "Number of Years to Compare (including selected year):",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            help="Select how many past years of data for the same contract month to display."
        )

        if selected_reference_contract_symbol:
            match = re.match(r"([A-Z]+)([FGHJKMNQUVXZ])(\d{1,2})", selected_reference_contract_symbol)
            if not match:
                st.error(f"Invalid reference contract symbol format: {selected_reference_contract_symbol}")
            else:
                _, month_code, year_suffix_str = match.groups()
                base_year_2digit = int(year_suffix_str)
                
                current_century = (datetime.utcnow().year // 100) * 100
                if base_year_2digit <= (datetime.utcnow().year % 100) + 1:
                    base_year_full = current_century + base_year_2digit
                else:
                    base_year_full = (current_century - 100) + base_year_2digit
                
                comparison_data = []
                for i in range(num_years_to_compare):
                    comp_year_full = base_year_full - i
                    table_name = f"{PRODUCT_SYMBOL}{FUTURES_MONTH_CODES_REV[month_code]:02d}{comp_year_full}".lower()
                    
                    df_contract_history = get_contract_data_from_db(table_name) # This will reuse the engine

                    if not df_contract_history.empty:
                        df_contract_history['Year'] = comp_year_full
                        
                        df_contract_history['Normalized Date'] = df_contract_history.index.map(lambda d: 
                            d.replace(year=base_year_full) if not (d.month == 2 and d.day == 29 and not calendar.isleap(base_year_full))
                            else d.replace(year=base_year_full, day=28) 
                        )
                        
                        df_contract_history['open_interest'] = pd.to_numeric(df_contract_history['open_interest'], errors='coerce')

                        df_contract_history_filtered = df_contract_history[df_contract_history['open_interest'].notna() & (df_contract_history['open_interest'] != 0)].copy()

                        if not df_contract_history_filtered.empty:
                            comparison_data.append(df_contract_history_filtered.reset_index()) 
                        else:
                            st.info(f"No valid (non-zero) OI data for {PRODUCT_SYMBOL}{month_code}{comp_year_full%100:02d} for historical comparison.")

                    else:
                        st.info(f"No data found for {PRODUCT_SYMBOL}{month_code}{comp_year_full%100:02d} (table '{table_name}') for historical comparison.")

                if comparison_data:
                    df_all_years = pd.concat(comparison_data)
                    df_all_years = df_all_years.sort_values(by=['Normalized Date', 'Year'])

                    fig_history = px.line(df_all_years, 
                                        x="Normalized Date", 
                                        y="open_interest", 
                                        color="Year",
                                        title=f"Historical Open Interest for {PRODUCT_SYMBOL}{month_code} Contracts (Normalized)", 
                                        labels={"open_interest": "Open Interest", "Normalized Date": f"Date (Normalized to {base_year_full})"},
                                        hover_data={"trade_date": "|%Y-%m-%d", "open_interest": ":,.0f", "Year": True})
                    
                    fig_history.update_layout(hovermode="x unified")
                    fig_history.update_yaxes(rangemode="tozero")

                    st.plotly_chart(fig_history, use_container_width=True)
                else:
                    st.info("No historical data found for the selected contract month and years to compare that has non-zero Open Interest.")

    st.markdown("---") 
    st.write("Data sourced from Databento via your ingestion script.")