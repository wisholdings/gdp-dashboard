import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import re
from dateutil.relativedelta import relativedelta
import calendar
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

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Future Contracts", page_icon="🔮", layout="wide")

# --- PAGE CONTENT ---
st.title("🔮 Future Contracts Analysis")
st.markdown("---")

st.markdown("""
**Compare active contracts from the same delivery month across different years using time-to-expiry analysis.**

This page allows you to overlay future contracts to identify trading opportunities, compare current market behavior 
against historical patterns, and analyze cross-year arbitrage potential.
""")

# Sidebar reference
with st.sidebar:
    st.title("🧭 Navigation")
    
    # Navigation buttons
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("streamlit_app.py")
    
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
    st.info("📍 **Current Page:** Future Contracts")
    
    st.subheader("📅 Futures Month Codes:")
    for month_num, code in FUTURES_MONTH_CODES.items():
        st.write(f"**{code}**: {calendar.month_name[month_num]}")
    
    st.markdown("---")
    st.subheader("🎯 Analysis Tips:")
    st.markdown("""
    - Compare contracts at similar lifecycle stages
    - Look for unusual OI patterns vs historical norms
    - Identify price divergences between years
    - Monitor liquidity differences across contracts
    """)

# Establish DB connection
engine = get_db_engine()
if not engine: 
    st.stop()

# Select Future Contract Only
future_contracts = get_future_contracts_only()

if not future_contracts:
    st.warning("⚠️ No future Natural Gas futures contracts found in the database. Please ensure your ingestion script has populated data for upcoming contracts.")
else:
    # Create display options
    future_contract_options = [f"{contract['symbol']} (Exp: {contract['expiry_date']})" for contract in future_contracts]
    
    selected_future_option = st.selectbox(
        "🎯 Select a Future Futures Contract:", 
        options=future_contract_options,
        help="Choose a future Natural Gas futures contract to view alongside other contracts from the same delivery month."
    )

    if selected_future_option:
        # Find the selected contract
        selected_future_index = future_contract_options.index(selected_future_option)
        selected_future_contract = future_contracts[selected_future_index]
        
        st.subheader(f"📈 Future Overlay for {calendar.month_name[selected_future_contract['month_num']]} Delivery Contracts")
        
        # Get all future contracts for the same month
        same_month_future_contracts = get_future_contracts_for_same_month(
            selected_future_contract['month_num'], 
            selected_future_contract['year']
        )
        
        if len(same_month_future_contracts) < 2:
            st.warning(f"⚠️ Only one future contract found for {calendar.month_name[selected_future_contract['month_num']]} delivery. Need multiple years to create overlay.")
        else:
            # Show contract count and time info
            nearest_expiry = min(contract['expiry_date'] for contract in same_month_future_contracts)
            days_to_nearest = (nearest_expiry - datetime.now().date()).days
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📊 Displaying {len(same_month_future_contracts)} active contracts for {calendar.month_name[selected_future_contract['month_num']]} delivery")
            with col2:
                st.info(f"⏰ Nearest expiry in {days_to_nearest} days ({nearest_expiry})")
            
            # Create overlay plots
            fig_future_oi = go.Figure()
            fig_future_settlement = go.Figure()
            
            # Define colors manually to avoid plotly express import
            colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4']
            
            for i, contract in enumerate(same_month_future_contracts):
                df_contract = get_contract_data_from_db(contract['table_name'])
                
                if not df_contract.empty:
                    # Calculate days to expiry
                    df_contract['days_to_expiry'] = calculate_days_to_expiry(
                        df_contract['trade_date'], 
                        contract['expiry_date']
                    )
                    
                    # Filter valid data (only include data before expiry)
                    df_oi = df_contract[
                        df_contract['open_interest'].notna() & 
                        (df_contract['open_interest'] != 0) &
                        (df_contract['days_to_expiry'] >= 0)
                    ].copy()
                    
                    df_settlement = df_contract[
                        df_contract['settlement_price'].notna() &
                        (df_contract['days_to_expiry'] >= 0)
                    ].copy()
                    
                    color = colors[i % len(colors)]
                    contract_label = f"{contract['symbol']} ({contract['year']})"
                    
                    # Add emphasis for current year contracts
                    line_width = 3 if contract['year'] == datetime.now().year else 2
                    line_dash = 'solid' if contract['year'] == datetime.now().year else 'dash'
                    
                    # Add Open Interest trace
                    if not df_oi.empty:
                        fig_future_oi.add_trace(go.Scatter(
                            x=df_oi['days_to_expiry'],
                            y=df_oi['open_interest'],
                            mode='lines',
                            name=contract_label,
                            line=dict(color=color, width=line_width, dash=line_dash),
                            hovertemplate=f"{contract_label}<br>" +
                                        "Days to Expiry: %{x}<br>" +
                                        "Open Interest: %{y:,.0f}<br>" +
                                        "<extra></extra>"
                        ))
                    
                    # Add Settlement Price trace
                    if not df_settlement.empty:
                        fig_future_settlement.add_trace(go.Scatter(
                            x=df_settlement['days_to_expiry'],
                            y=df_settlement['settlement_price'],
                            mode='lines',
                            name=contract_label,
                            line=dict(color=color, width=line_width, dash=line_dash),
                            hovertemplate=f"{contract_label}<br>" +
                                        "Days to Expiry: %{x}<br>" +
                                        "Settlement Price: $%{y:.2f}<br>" +
                                        "<extra></extra>"
                        ))
            
            # Update layout for Open Interest
            fig_future_oi.update_layout(
                title=f"Open Interest Overlay - Future {calendar.month_name[selected_future_contract['month_num']]} Delivery Contracts",
                xaxis_title="Days to Expiry",
                yaxis_title="Open Interest",
                hovermode='closest',
                xaxis=dict(autorange='reversed'),  # Reverse so expiry (0) is on the right
                height=500,
                showlegend=True
            )
            
            # Update layout for Settlement Price
            fig_future_settlement.update_layout(
                title=f"Settlement Price Overlay - Future {calendar.month_name[selected_future_contract['month_num']]} Delivery Contracts",
                xaxis_title="Days to Expiry",
                yaxis_title="Settlement Price ($)",
                hovermode='closest',
                xaxis=dict(autorange='reversed'),  # Reverse so expiry (0) is on the right
                height=500,
                showlegend=True
            )
            
            # Display plots
            st.plotly_chart(fig_future_oi, use_container_width=True)
            st.plotly_chart(fig_future_settlement, use_container_width=True)
            
            # Show summary
            st.subheader("📋 Future Contract Summary")
            future_summary_data = []
            for contract in same_month_future_contracts:
                df_contract = get_contract_data_from_db(contract['table_name'])
                if not df_contract.empty:
                    # Get most recent non-null values instead of just the last row
                    valid_oi = df_contract['open_interest'].dropna()
                    valid_price = df_contract['settlement_price'].dropna()
                    
                    current_oi = valid_oi.iloc[-1] if not valid_oi.empty else 0
                    current_price = valid_price.iloc[-1] if not valid_price.empty else 0
                    data_points = len(df_contract)
                    days_remaining = (contract['expiry_date'] - datetime.now().date()).days
                    
                    future_summary_data.append({
                        'Contract': contract['symbol'],
                        'Year': contract['year'],
                        'Expiry Date': contract['expiry_date'],
                        'Days Remaining': days_remaining,
                        'Data Points': data_points,
                        'Current Open Interest': f"{current_oi:,.0f}" if current_oi > 0 else "N/A",
                        'Current Settlement Price': f"${current_price:.2f}" if current_price > 0 else "N/A"
                    })
            
            future_summary_df = pd.DataFrame(future_summary_data)
            st.dataframe(future_summary_df, use_container_width=True)
            
            # Trading insights
            st.subheader("💡 Trading Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🔄 Cross-Year Arbitrage:**
                - Compare prices between years at similar days-to-expiry
                - Look for unusual spreads that may present opportunities
                - Monitor convergence patterns as expiry approaches
                """)
            
            with col2:
                st.markdown("""
                **📊 Liquidity Analysis:**
                - Higher OI typically indicates better liquidity
                - Compare current OI levels to historical norms
                - Identify contracts with building vs declining interest
                """)
            
            # Current market status
            current_year_contracts = [c for c in same_month_future_contracts if c['year'] == datetime.now().year]
            if current_year_contracts:
                st.info(f"💡 **Current Year Focus**: {datetime.now().year} contracts are highlighted with solid lines in the charts above.")

st.markdown("---")
st.markdown("**💾 Data Source:** Databento via automated ingestion script | **🔄 Data Updates:** Every hour")