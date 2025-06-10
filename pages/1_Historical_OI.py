import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import calendar

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
        st.error(f"Error connecting to database: {e}")
        st.stop() 

@st.cache_data(ttl=3600)
def get_power_burns_data(start_date=None, end_date=None):
    """Fetch power burns data from the database"""
    engine = get_db_engine()
    df = pd.DataFrame()
    
    try:
        # Get today's date for filtering
        today = date.today()
        
        # Base query - filter by today's date_published
        base_query = f"""
            SELECT report_date, L48_Power_Burns, date_published 
            FROM power_burns_daily 
            WHERE DATE(date_published) = '{today}'
        """
        
        # Add date filtering if provided
        if start_date and end_date:
            query = f"""
                {base_query}
                AND report_date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY report_date ASC
            """
        else:
            query = f"{base_query} ORDER BY report_date ASC"
        
        df = pd.read_sql_query(query, engine)
        
        if not df.empty:
            df['report_date'] = pd.to_datetime(df['report_date']).dt.date
            df['date_published'] = pd.to_datetime(df['date_published'])
            df['L48_Power_Burns'] = pd.to_numeric(df['L48_Power_Burns'], errors='coerce')
            
            # Add derived columns for analysis
            df['year'] = pd.to_datetime(df['report_date']).dt.year
            df['month'] = pd.to_datetime(df['report_date']).dt.month
            df['day_of_year'] = pd.to_datetime(df['report_date']).dt.dayofyear
            df['quarter'] = pd.to_datetime(df['report_date']).dt.quarter
            
        return df
        
    except Exception as e:
        st.error(f"Error fetching power burns data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_data_date_range():
    """Get the available date range for power burns data"""
    engine = get_db_engine()
    today = date.today()
    try:
        query = f"""
            SELECT MIN(report_date) as min_date, MAX(report_date) as max_date 
            FROM power_burns_daily 
            WHERE DATE(date_published) = '{today}'
        """
        result = pd.read_sql_query(query, engine)
        if not result.empty and result['min_date'].iloc[0] is not None:
            min_date = pd.to_datetime(result['min_date'].iloc[0]).date()
            max_date = pd.to_datetime(result['max_date'].iloc[0]).date()
            return min_date, max_date
    except Exception as e:
        st.error(f"Error getting date range: {e}")
    return None, None

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Power Burns Analysis", page_icon="🔥", layout="wide")

# --- PAGE CONTENT ---
st.title("🔥 Natural Gas Power Burns Analysis")
st.markdown("---")

# Get today's date for display
today = date.today()

st.markdown(f"""
**Analyze daily natural gas consumption for power generation in the Lower 48 states.**

**📅 Data Filter:** This page shows **ONLY today's forecast data** (date_published = {today})

This page provides comprehensive analysis of power burns data, including seasonal patterns, 
yearly trends, and forecast data visualization.
""")

# Sidebar controls
with st.sidebar:
    st.title("🧭 Navigation")
    
    # Navigation buttons
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("streamlit_app.py")
    
    if st.button("📊 Historical OI", use_container_width=True):
        st.switch_page("pages/1_Historical_OI.py")
        
    if st.button("🔮 Future Contracts", use_container_width=True):
        st.switch_page("pages/2_Future_Contracts.py")
        
    if st.button("⚡ EIA Generation", use_container_width=True):
        st.switch_page("pages/3_EIA_Generation.py")
        
    if st.button("📈 Net Changes", use_container_width=True):
        st.switch_page("pages/5_Net_Changes.py")
        
    if st.button("📊 Tape Analysis", use_container_width=True):
        st.switch_page("pages/6_Tape_Analysis.py")
    
    st.markdown("---")
    st.info("📍 **Current Page:** Power Burns")
    
    # Show today's date prominently
    st.success(f"📅 **Data Filter:** {today}")
    st.caption("Only showing today's published forecast data")
    
    st.subheader("📊 Analysis Controls")
    
    # Get available date range
    min_date, max_date = get_data_date_range()
    
    if min_date and max_date:
        st.write(f"**Available Forecast Period:** {min_date} to {max_date}")
        
        # Date range selection for forecast periods
        st.subheader("📅 Forecast Period Filter")
        st.caption("Filter which forecast periods to analyze")
        
        # Default to next 30 days of forecast
        default_start = min_date
        default_end = min(max_date, min_date + timedelta(days=30))
        
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.date_input(
            "End Date", 
            value=default_end,
            min_value=min_date,
            max_value=max_date
        )
        
        if start_date > end_date:
            st.error("Start date must be before end date")
            st.stop()
        
        # Chart options
        st.subheader("📈 Chart Options")
        show_trend = st.checkbox("Show Trend Line", value=True)
        show_moving_avg = st.checkbox("Show Moving Average", value=False)
        if show_moving_avg:
            ma_days = st.slider("Moving Average Days:", 7, 30, 14)
    else:
        st.error(f"Unable to determine forecast date range for today's data ({today})")
        st.warning("This might mean today's forecast data hasn't been uploaded yet.")
        st.stop()

# Establish DB connection
engine = get_db_engine()
if not engine: 
    st.stop()

# Main content area
if min_date and max_date:
    
    # Load data
    with st.spinner(f"Loading today's power burns forecast data ({today})..."):
        df = get_power_burns_data(start_date, end_date)
    
    if df.empty:
        st.warning(f"No power burns forecast data found for {today} in the selected period ({start_date} to {end_date}).")
        st.info("💡 This might mean today's forecast data hasn't been uploaded yet.")
    else:
        # Display data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Points", f"{len(df):,}")
        with col2:
            avg_burns = df['L48_Power_Burns'].mean()
            st.metric("Average Burns", f"{avg_burns:,.1f} Bcf/d")
        with col3:
            peak_burns = df['L48_Power_Burns'].max()
            st.metric("Peak Burns", f"{peak_burns:,.1f} Bcf/d")
        with col4:
            st.metric("Published", str(today))
        
        # Show forecast info
        forecast_start = df['report_date'].min()
        forecast_end = df['report_date'].max()
        st.info(f"📊 Showing forecast from {forecast_start} to {forecast_end} (published on {today})")
        
        # Create time series plot
        fig = go.Figure()
        
        # Main data line
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df['report_date']),
            y=df['L48_Power_Burns'],
            mode='lines',
            name='Power Burns',
            line=dict(color='#ff6b35', width=2),
            hovertemplate="Date: %{x}<br>Power Burns: %{y:.1f} Bcf/d<extra></extra>"
        ))
        
        # Add trend line if requested
        if show_trend and len(df) > 1:
            z = np.polyfit(range(len(df)), df['L48_Power_Burns'].dropna(), 1)
            trend_line = np.poly1d(z)(range(len(df)))
            
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(df['report_date']),
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate="Trend: %{y:.1f} Bcf/d<extra></extra>"
            ))
        
        # Add moving average if requested
        if show_moving_avg and len(df) > ma_days:
            df_sorted = df.sort_values('report_date')
            ma_values = df_sorted['L48_Power_Burns'].rolling(window=ma_days, center=True).mean()
            
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(df_sorted['report_date']),
                y=ma_values,
                mode='lines',
                name=f'{ma_days}-Day MA',
                line=dict(color='blue', width=2),
                hovertemplate=f"{ma_days}-Day MA: %{{y:.1f}} Bcf/d<extra></extra>"
            ))
        
        fig.update_layout(
            title=f"Daily Natural Gas Power Burns Forecast - {today} Publication",
            xaxis_title="Date",
            yaxis_title="Power Burns (Bcf/d)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Min Burns", f"{df['L48_Power_Burns'].min():.1f} Bcf/d")
            st.metric("Max Burns", f"{df['L48_Power_Burns'].max():.1f} Bcf/d")
        
        with col2:
            st.metric("Average Burns", f"{df['L48_Power_Burns'].mean():.1f} Bcf/d")
            st.metric("Std Deviation", f"{df['L48_Power_Burns'].std():.1f} Bcf/d")
        
        with col3:
            st.metric("Total Period", f"{len(df)} days")
            date_span = (end_date - start_date).days + 1
            st.metric("Selected Range", f"{date_span} days")
        
        # Raw data section
        with st.expander("📋 Raw Data Sample"):
            # Show sample with key columns
            display_cols = ['report_date', 'L48_Power_Burns', 'date_published']
            st.dataframe(df[display_cols].head(50), use_container_width=True)
            
            if st.button("📥 Download Full Dataset"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"power_burns_forecast_{today}_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )

st.markdown("---")
st.markdown(f"**💾 Data Source:** EIA Power Burns Data | **🔄 Data Updates:** Daily | **📅 Current Filter:** {today} forecast data only")