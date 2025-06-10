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
    page_title="Energy Analysis Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
with st.sidebar:
    st.title("🔋 Energy Analysis")
    st.markdown("---")
    
    # Manual Navigation Buttons
    st.subheader("🧭 Navigation")
    
    if st.button("📊 Historical Open Interest", use_container_width=True):
        st.switch_page("pages/1_Historical_OI.py")
    
    if st.button("🔮 Future Contracts", use_container_width=True):
        st.switch_page("pages/2_Future_Contracts.py")
        
    if st.button("⚡ EIA Generation", use_container_width=True):
        st.switch_page("pages/3_EIA_Generation.py")

    if st.button("🔥 Power Burns", use_container_width=True):
        st.switch_page("pages/4_Power_Burns.py")

    if st.button("📈 Net Changes", use_container_width=True):
        st.switch_page("pages/5_Net_Changes.py")
    
    if st.button("📊 Tape Analysis", use_container_width=True):
        st.switch_page("pages/6_Tape_Analysis.py")
    
    st.markdown("---")
    
    # Current page indicator
    st.info("📍 **Current Page:** Home")
    
    st.markdown("""
    **💡 Navigation Tips:**
    • Click the buttons above to switch pages
    • Each page has specialized analysis tools
    • Data updates hourly automatically
    • Use the Home button in each page to return here
    """)
    
    # Quick reference
    st.markdown("---")
    st.subheader("📅 NG Month Codes")
    month_codes = {
        "F": "Jan", "G": "Feb", "H": "Mar", "J": "Apr",
        "K": "May", "M": "Jun", "N": "Jul", "Q": "Aug", 
        "U": "Sep", "V": "Oct", "X": "Nov", "Z": "Dec"
    }
    
    for code, month in month_codes.items():
        st.write(f"**{code}** = {month}")

st.title("⚡ Energy Analysis Platform")
st.markdown("---")

# Navigation instructions
st.success("""
🧭 **How to Navigate:** Use the **navigation buttons** in the left sidebar to access different analysis tools. 
Click any button to switch to that page's specialized functionality.
""")

# Quick access buttons in main area too
st.subheader("🚀 Quick Access")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📊 Go to Historical OI Analysis", use_container_width=True):
        st.switch_page("pages/1_Historical_OI.py")
    st.write("Analyze expired NG contracts using time-to-expiry overlays")

with col2:
    if st.button("🔮 Go to Future Contracts", use_container_width=True):
        st.switch_page("pages/2_Future_Contracts.py")
    st.write("Compare active contracts and identify arbitrage opportunities")

with col3:
    if st.button("⚡ Go to Generation Analysis", use_container_width=True):
        st.switch_page("pages/3_EIA_Generation.py")
    st.write("Explore hourly electricity generation by source and region")

with col4:
    if st.button("🔥 Go to Power Burns Analysis", use_container_width=True):
        st.switch_page("pages/4_Power_Burns.py")
    st.write("Analyze natural gas power consumption forecasts")

with col5:
    if st.button("📈 Go to Net Changes Analysis", use_container_width=True):
        st.switch_page("pages/5_Net_Changes.py")
    st.write("Analyze day-over-day changes across all regions")

st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Welcome to the Energy Analysis Platform
    
    This application provides comprehensive analysis tools for energy markets:
    
    ### 📊 Analysis Modules:
    
    **📈 Natural Gas Futures Analysis:**
    - **📊 Historical Open Interest** - Time-to-expiry overlays for expired contracts
    - **🔮 Future Contracts** - Active contract comparisons and arbitrage analysis
    
    **⚡ Electricity Generation Analysis:**
    - **⚡ EIA Generation** - Hourly electricity generation by source and region
    - **🔥 Power Burns** - Natural gas consumption for power generation analysis
    - **📈 Net Changes** - Day-over-day forecast changes across all regions
    
    **📊 Market Microstructure:**
    - **📊 Tape Analysis** - High-frequency trading data analysis
    
    ### 🚀 Getting Started:
    1. **Click on a page** in the sidebar navigation above ⬆️
    2. **Select your analysis type** (Historical, Future, or Generation)
    3. **Choose contracts or regions** using the page controls
    4. **Analyze the interactive charts** and download data as needed
    """)

with col2:
    st.success("""
    **📋 Platform Features:**
    
    ✅ Real-time database connectivity  
    ✅ Hourly automated data updates  
    ✅ Interactive time-series visualizations  
    ✅ Cross-year contract comparisons  
    ✅ Generation source breakdowns  
    ✅ Forecast comparison analysis  
    ✅ High-frequency tape analysis  
    ✅ Export capabilities  
    ✅ Time-to-expiry analysis  
    """)

# Database connection check
st.markdown("---")
st.subheader("🔗 System Status")

try:
    engine = get_db_engine()
    if engine:
        # Get Natural Gas contracts stats
        all_ng_tables = get_all_contract_table_names(PRODUCT_SYMBOL)
        historical_count = len(get_historical_contracts_only())
        future_count = len(get_future_contracts_only())
        
        # Get generation tables stats
        inspector = inspect(engine)
        all_table_names = inspector.get_table_names()
        generation_tables = [t for t in all_table_names if t.endswith('_hourly_generation')]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Database Status", "✅ Connected")
        with col2:
            st.metric("NG Contracts", len(all_ng_tables))
        with col3:
            st.metric("Historical", historical_count)
        with col4:
            st.metric("Future", future_count)
        with col5:
            st.metric("Generation Tables", len(generation_tables))
            
        # Show available generation regions
        if generation_tables:
            st.markdown("### 🗺️ Available Generation Regions:")
            regions = [t.replace('_hourly_generation', '').replace('_', ' ').title() for t in generation_tables]
            st.write(", ".join(regions))
            
except Exception as e:
    st.error(f"❌ Database connection failed: {e}")

# Quick Market Overview
st.markdown("---")
st.subheader("📈 Quick Market Overview")

try:
    # Get some recent data for display
    recent_contracts = get_future_contracts_only()[:3]  # Next 3 expiring contracts
    
    if recent_contracts:
        st.markdown("**🔮 Next Expiring NG Contracts:**")
        for contract in recent_contracts:
            days_remaining = (contract['expiry_date'] - datetime.now().date()).days
            st.write(f"• **{contract['symbol']}** expires in {days_remaining} days ({contract['expiry_date']})")
    
    # Show generation data availability
    if generation_tables:
        st.markdown("**⚡ Generation Data Coverage:**")
        sample_table = generation_tables[0]  # Check first table for date range
        try:
            sample_query = f"SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM `{sample_table}`"
            with engine.connect() as conn:
                result = conn.execute(text(sample_query)).fetchone()
                if result:
                    min_date = result[0]
                    max_date = result[1]
                    if min_date and max_date:
                        st.write(f"• Data available from {min_date.date()} to {max_date.date()}")
        except:
            st.write("• Generation data tables available")
            
except Exception as e:
    st.write("Unable to load quick stats")

st.markdown("---")
st.markdown("**💾 Data Sources:** Databento (NG Futures) | EIA (Generation) | **🔄 Updates:** Every hour")
import os
st.write("Files in pages directory:")
try:
    files = os.listdir("pages")
    st.write(files)
except Exception as e:
    st.write(f"Error listing files: {e}")

st.write("Power Burns exists:", os.path.exists("pages/4_Power_Burns.py"))