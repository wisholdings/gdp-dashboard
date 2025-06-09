import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import re
from dateutil.relativedelta import relativedelta
import calendar
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os 

# Import SQLAlchemy components
from sqlalchemy import create_engine, inspect, text 
from sqlalchemy.exc import OperationalError, ProgrammingError 
import mysql.connector 

# --- HARDCODED DATABASE CREDENTIALS (FOR TESTING ONLY) ---
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

@st.cache_resource 
def get_db_engine():
    try:
        connection_string = (
            f"mysql+mysqlconnector://{HARDCODED_DB_USER}:"
            f"{HARDCODED_DB_PASSWORD}@{HARDCODED_DB_HOST}:"
            f"{HARDCODED_DB_PORT}/{HARDCODED_DB_DATABASE}"
        )
        
        engine = create_engine(connection_string, echo=False) 
        
        with engine.connect() as connection:
            connection.execute(text("SELECT 1")).scalar() 
        
        return engine
    except Exception as e:
        st.error(f"Error connecting to database. Please ensure your Cloud SQL instance is running, "
                 f"its public IP is correct, and that the environment's outbound IP is authorized (`0.0.0.0/0` if on App Engine). "
                 f"Also check database user/password are correct. Full error: {e}")
        st.stop() 

def _get_last_day_of_month(year, month):
    return calendar.monthrange(year, month)[1]

def get_contract_expiry_date(year, month):
    """Calculate the expiry date for a natural gas contract (last day of month before delivery month)"""
    exp_year = year
    exp_month = month - 1
    if exp_month == 0:
        exp_month = 12
        exp_year -= 1
    
    try:
        expiration_day = calendar.monthrange(exp_year, exp_month)[1]
        return datetime(exp_year, exp_month, expiration_day).date()
    except ValueError:
        return datetime(year, month, 1).date()

@st.cache_data(ttl=3600)
def generate_forward_contract_symbols(start_dt: datetime, num_months_out: int):
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
            expiration_day = calendar.monthrange(exp_year_cand, exp_month_cand)[1]
            expiration_datetime_obj = datetime(exp_year_cand, exp_month_cand, expiration_day)
        except ValueError:
            expiration_datetime_obj = datetime(contract_y, contract_m, 1) 
        
        generated_contracts.append((symbol_name, contract_y, contract_m, expiration_datetime_obj))
        current_processing_date += relativedelta(months=1)
        
    return generated_contracts

@st.cache_data(ttl=3600)
def get_all_contract_table_names(product_symbol="ng"):
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
    engine = get_db_engine()
    df = pd.DataFrame()
    try:
        query = f"SELECT trade_date, open_interest, settlement_price FROM `{table_name.lower()}` ORDER BY trade_date ASC"
            
        df = pd.read_sql_query(query, engine)
        
        if not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
            df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce')
            df['settlement_price'] = pd.to_numeric(df['settlement_price'], errors='coerce')
        
    except Exception as e:
        st.error(f"Error fetching data for table '{table_name}': {e}")
    return df

def parse_contract_symbol(symbol):
    """Parse a contract symbol and return product, month_code, year, and full year"""
    match = re.match(r"([A-Z]+)([FGHJKMNQUVXZ])(\d{1,2})", symbol)
    if not match:
        return None, None, None, None
    
    product, month_code, year_suffix_str = match.groups()
    base_year_2digit = int(year_suffix_str)
    
    # Convert 2-digit year to 4-digit year
    current_century = (datetime.utcnow().year // 100) * 100
    if base_year_2digit <= (datetime.utcnow().year % 100) + 1: 
        full_year = current_century + base_year_2digit
    else: 
        full_year = (current_century - 100) + base_year_2digit
    
    month_num = FUTURES_MONTH_CODES_REV[month_code]
    
    return product, month_code, month_num, full_year

def get_historical_contracts_only():
    """Get only contracts that have already expired"""
    all_futures_table_names = get_all_contract_table_names(PRODUCT_SYMBOL)
    historical_contracts = []
    current_date = datetime.now().date()
    
    for table_name in all_futures_table_names:
        match = re.match(rf"^{PRODUCT_SYMBOL.lower()}(\d{{2}})(\d{{4}})$", table_name)
        if match:
            month_num = int(match.group(1))
            year_full = int(match.group(2))
            month_code = FUTURES_MONTH_CODES.get(month_num)
            
            if month_code:
                # Calculate expiry date
                expiry_date = get_contract_expiry_date(year_full, month_num)
                
                # Only include if contract has already expired
                if expiry_date < current_date:
                    display_symbol = f"{PRODUCT_SYMBOL}{month_code}{year_full%100:02d}"
                    historical_contracts.append({
                        'symbol': display_symbol,
                        'month_num': month_num,
                        'year': year_full,
                        'expiry_date': expiry_date,
                        'table_name': table_name
                    })
    
    return sorted(historical_contracts, key=lambda x: x['expiry_date'], reverse=True)

def get_contracts_for_same_month(target_month_num, target_year):
    """Get all historical contracts for the same delivery month"""
    all_futures_table_names = get_all_contract_table_names(PRODUCT_SYMBOL)
    same_month_contracts = []
    current_date = datetime.now().date()
    
    for table_name in all_futures_table_names:
        match = re.match(rf"^{PRODUCT_SYMBOL.lower()}(\d{{2}})(\d{{4}})$", table_name)
        if match:
            month_num = int(match.group(1))
            year_full = int(match.group(2))
            month_code = FUTURES_MONTH_CODES.get(month_num)
            
            if month_code and month_num == target_month_num:
                expiry_date = get_contract_expiry_date(year_full, month_num)
                
                # Only include if contract has already expired
                if expiry_date < current_date:
                    display_symbol = f"{PRODUCT_SYMBOL}{month_code}{year_full%100:02d}"
                    same_month_contracts.append({
                        'symbol': display_symbol,
                        'month_num': month_num,
                        'year': year_full,
                        'expiry_date': expiry_date,
                        'table_name': table_name
                    })
    
    return sorted(same_month_contracts, key=lambda x: x['year'])

def get_future_contracts_only():
    """Get only contracts that have not yet expired"""
    all_futures_table_names = get_all_contract_table_names(PRODUCT_SYMBOL)
    future_contracts = []
    current_date = datetime.now().date()
    
    for table_name in all_futures_table_names:
        match = re.match(rf"^{PRODUCT_SYMBOL.lower()}(\d{{2}})(\d{{4}})$", table_name)
        if match:
            month_num = int(match.group(1))
            year_full = int(match.group(2))
            month_code = FUTURES_MONTH_CODES.get(month_num)
            
            if month_code:
                # Calculate expiry date
                expiry_date = get_contract_expiry_date(year_full, month_num)
                
                # Only include if contract has not yet expired
                if expiry_date >= current_date:
                    display_symbol = f"{PRODUCT_SYMBOL}{month_code}{year_full%100:02d}"
                    future_contracts.append({
                        'symbol': display_symbol,
                        'month_num': month_num,
                        'year': year_full,
                        'expiry_date': expiry_date,
                        'table_name': table_name
                    })
    
    return sorted(future_contracts, key=lambda x: x['expiry_date'])

def get_future_contracts_for_same_month(target_month_num, target_year):
    """Get all future contracts for the same delivery month"""
    all_futures_table_names = get_all_contract_table_names(PRODUCT_SYMBOL)
    same_month_contracts = []
    current_date = datetime.now().date()
    
    for table_name in all_futures_table_names:
        match = re.match(rf"^{PRODUCT_SYMBOL.lower()}(\d{{2}})(\d{{4}})$", table_name)
        if match:
            month_num = int(match.group(1))
            year_full = int(match.group(2))
            month_code = FUTURES_MONTH_CODES.get(month_num)
            
            if month_code and month_num == target_month_num:
                expiry_date = get_contract_expiry_date(year_full, month_num)
                
                # Only include if contract has not yet expired
                if expiry_date >= current_date:
                    display_symbol = f"{PRODUCT_SYMBOL}{month_code}{year_full%100:02d}"
                    same_month_contracts.append({
                        'symbol': display_symbol,
                        'month_num': month_num,
                        'year': year_full,
                        'expiry_date': expiry_date,
                        'table_name': table_name
                    })
    
    return sorted(same_month_contracts, key=lambda x: x['year'])

def calculate_days_to_expiry(trade_dates, expiry_date):
    """Calculate days to expiry for each trade date"""
    return [(expiry_date - trade_date).days for trade_date in trade_dates]

# --- MAIN APP HOMEPAGE ---

# --- 0. SIMPLE AUTHENTICATION STATE ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = True 

if not st.session_state['authenticated']:
    st.title("Login Required (Auth Disabled for Direct Test)")
    st.write("Authentication is temporarily disabled in this version for direct DB connection testing.")
    st.stop()

# Main page configuration
st.set_page_config(
    page_title="NG Futures Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔥 Natural Gas Futures Analysis Platform")
st.markdown("---")

# Welcome section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Welcome to the NG Futures Analysis Platform
    
    This application provides comprehensive analysis tools for Natural Gas futures contracts:
    
    ### 📊 Available Analysis Tools:
    
    **🕰️ Historical Open Interest** - Analyze expired contracts using time-to-expiry overlays to identify patterns across years for the same delivery month.
    
    **🔮 Future Contracts** - Compare active contracts from the same delivery month across different years to spot opportunities and trends.
    
    ### 🚀 Getting Started:
    Use the sidebar navigation to access different analysis pages. Each page provides specialized tools for understanding Natural Gas futures market dynamics.
    """)

with col2:
    st.info("""
    **📋 Quick Tips:**
    
    • Data refreshes hourly from the database
    • Time-to-expiry analysis reveals seasonal patterns
    • Compare contracts across multiple years
    • Export data for further analysis
    """)

# Database connection check
st.markdown("---")
st.subheader("🔗 System Status")

try:
    engine = get_db_engine()
    if engine:
        # Get some basic stats
        all_tables = get_all_contract_table_names(PRODUCT_SYMBOL)
        historical_count = len(get_historical_contracts_only())
        future_count = len(get_future_contracts_only())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Database Status", "✅ Connected")
        with col2:
            st.metric("Total Contracts", len(all_tables))
        with col3:
            st.metric("Historical Contracts", historical_count)
        with col4:
            st.metric("Future Contracts", future_count)
            
except Exception as e:
    st.error(f"❌ Database connection failed: {e}")

# Futures month codes reference
st.markdown("---")
st.subheader("📅 Futures Month Codes Reference")

cols = st.columns(6)
months_per_col = 2

for i, (month_num, code) in enumerate(FUTURES_MONTH_CODES.items()):
    col_idx = i // months_per_col
    with cols[col_idx]:
        st.write(f"**{code}** - {calendar.month_name[month_num]}")

st.markdown("---")
st.markdown("**💾 Data Source:** Databento via automated ingestion script")
st.markdown("**⚡ Last Updated:** Data refreshes automatically every hour")