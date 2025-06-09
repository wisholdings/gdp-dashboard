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

def calculate_days_to_expiry(trade_dates, expiry_date):
    """Calculate days to expiry for each trade date"""
    return [(expiry_date - trade_date).days for trade_date in trade_dates]

# --- Streamlit App UI ---

# --- 0. SIMPLE AUTHENTICATION STATE ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = True 

if not st.session_state['authenticated']:
    st.title("Login Required (Auth Disabled for Direct Test)")
    st.write("Authentication is temporarily disabled in this version for direct DB connection testing.")
    st.stop()
else:
    st.set_page_config(page_title="NG Futures Historical Data")
    st.title("Natural Gas Futures Historical Contract Data")
    st.write("View historical contracts overlaid by time-to-expiry for the same delivery month.")

    st.info(f"Data in this app is refreshed from the database every hour using `@st.cache_data`. "
            f"Your ingestion script determines how frequently the database itself is updated.")

    st.sidebar.subheader("Futures Month Codes:")
    for month_num, code in FUTURES_MONTH_CODES.items():
        st.sidebar.write(f"**{code}**: {calendar.month_name[month_num]}")

    # --- Establish DB connection ---
    engine = get_db_engine()
    if not engine: 
        st.stop()

    # Create tabs
    tab1, tab2 = st.tabs(["Historical_OI", "Tab 2"])
    
    with tab1:
        st.subheader("Historical Open Interest Analysis")
        st.write("Compare contracts from the same delivery month across different years using time-to-expiry.")
        
        # --- Select Historical Contract Only ---
        historical_contracts = get_historical_contracts_only()
        
        if not historical_contracts:
            st.warning("No expired Natural Gas futures contracts found in the database. Please ensure your ingestion script has successfully populated historical data.")
        else:
            # Create display options
            contract_options = [f"{contract['symbol']} (Exp: {contract['expiry_date']})" for contract in historical_contracts]
            
            selected_option = st.selectbox(
                "Select a Historical Futures Contract:", 
                options=contract_options,
                help="Choose an expired Natural Gas futures contract to view alongside other contracts from the same delivery month."
            )

            if selected_option:
                # Find the selected contract
                selected_index = contract_options.index(selected_option)
                selected_contract = historical_contracts[selected_index]
                
                st.subheader(f"Historical Overlay for {calendar.month_name[selected_contract['month_num']]} Delivery Contracts")
                
                # Get all contracts for the same month
                same_month_contracts = get_contracts_for_same_month(
                    selected_contract['month_num'], 
                    selected_contract['year']
                )
                
                if len(same_month_contracts) < 2:
                    st.warning(f"Only one contract found for {calendar.month_name[selected_contract['month_num']]} delivery. Need multiple years to create overlay.")
                else:
                    # Create overlay plots
                    fig_oi = go.Figure()
                    fig_settlement = go.Figure()
                    
                    colors = px.colors.qualitative.Set1
                    
                    for i, contract in enumerate(same_month_contracts):
                        df_contract = get_contract_data_from_db(contract['table_name'])
                        
                        if not df_contract.empty:
                            # Calculate days to expiry
                            df_contract['days_to_expiry'] = calculate_days_to_expiry(
                                df_contract['trade_date'], 
                                contract['expiry_date']
                            )
                            
                            # Filter valid data
                            df_oi = df_contract[
                                df_contract['open_interest'].notna() & 
                                (df_contract['open_interest'] != 0) &
                                (df_contract['days_to_expiry'] >= 0)  # Only include data before expiry
                            ].copy()
                            
                            df_settlement = df_contract[
                                df_contract['settlement_price'].notna() &
                                (df_contract['days_to_expiry'] >= 0)  # Only include data before expiry
                            ].copy()
                            
                            color = colors[i % len(colors)]
                            contract_label = f"{contract['symbol']} ({contract['year']})"
                            
                            # Add Open Interest trace
                            if not df_oi.empty:
                                fig_oi.add_trace(go.Scatter(
                                    x=df_oi['days_to_expiry'],
                                    y=df_oi['open_interest'],
                                    mode='lines',
                                    name=contract_label,
                                    line=dict(color=color),
                                    hovertemplate=f"{contract_label}<br>" +
                                                "Days to Expiry: %{x}<br>" +
                                                "Open Interest: %{y:,.0f}<br>" +
                                                "<extra></extra>"
                                ))
                            
                            # Add Settlement Price trace
                            if not df_settlement.empty:
                                fig_settlement.add_trace(go.Scatter(
                                    x=df_settlement['days_to_expiry'],
                                    y=df_settlement['settlement_price'],
                                    mode='lines',
                                    name=contract_label,
                                    line=dict(color=color),
                                    hovertemplate=f"{contract_label}<br>" +
                                                "Days to Expiry: %{x}<br>" +
                                                "Settlement Price: $%{y:.2f}<br>" +
                                                "<extra></extra>"
                                ))
                    
                    # Update layout for Open Interest
                    fig_oi.update_layout(
                        title=f"Open Interest Overlay - {calendar.month_name[selected_contract['month_num']]} Delivery Contracts",
                        xaxis_title="Days to Expiry",
                        yaxis_title="Open Interest",
                        hovermode='closest',
                        xaxis=dict(autorange='reversed')  # Reverse so expiry (0) is on the right
                    )
                    
                    # Update layout for Settlement Price
                    fig_settlement.update_layout(
                        title=f"Settlement Price Overlay - {calendar.month_name[selected_contract['month_num']]} Delivery Contracts",
                        xaxis_title="Days to Expiry",
                        yaxis_title="Settlement Price ($)",
                        hovermode='closest',
                        xaxis=dict(autorange='reversed')  # Reverse so expiry (0) is on the right
                    )
                    
                    # Display plots
                    st.plotly_chart(fig_oi, use_container_width=True)
                    st.plotly_chart(fig_settlement, use_container_width=True)
                    
                    # Show summary
                    st.subheader("Contract Summary")
                    summary_data = []
                    for contract in same_month_contracts:
                        df_contract = get_contract_data_from_db(contract['table_name'])
                        if not df_contract.empty:
                            max_oi = df_contract['open_interest'].max() if df_contract['open_interest'].notna().any() else 0
                            avg_price = df_contract['settlement_price'].mean() if df_contract['settlement_price'].notna().any() else 0
                            data_points = len(df_contract)
                            
                            summary_data.append({
                                'Contract': contract['symbol'],
                                'Year': contract['year'],
                                'Expiry Date': contract['expiry_date'],
                                'Data Points': data_points,
                                'Max Open Interest': f"{max_oi:,.0f}" if max_oi > 0 else "N/A",
                                'Avg Settlement Price': f"${avg_price:.2f}" if avg_price > 0 else "N/A"
                            })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
    
    with tab2:
        st.subheader("Tab 2 Content")
        st.write("This is where you can add additional functionality for the second tab.")

    st.markdown("---") 
    st.write("Data sourced from Databento via your ingestion script.")