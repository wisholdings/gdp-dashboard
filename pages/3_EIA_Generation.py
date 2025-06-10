import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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
def get_generation_data(table_name: str, start_date=None, end_date=None):
    """Fetch generation data from the specified table - ONLY TODAY'S PUBLISHED DATA"""
    engine = get_db_engine()
    df = pd.DataFrame()
    
    try:
        # Get today's date
        today = date.today()
        
        # Base query - ALWAYS filter by today's date_published
        base_query = f"""
            SELECT * FROM `{table_name}` 
            WHERE DATE(date_published) = '{today}'
        """
        
        # Add timestamp filtering if provided (but keep date_published filter)
        if start_date and end_date:
            query = f"""
                {base_query}
                AND DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY timestamp ASC
            """
        else:
            query = f"{base_query} ORDER BY timestamp ASC"
        
        df = pd.read_sql_query(query, engine)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Convert numeric columns
            numeric_columns = [col for col in df.columns if col not in ['timestamp', 'date_published']]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # If no data for today, show a warning
            st.warning(f"⚠️ No data found for today's date_published ({today}) in table '{table_name}'. This might mean today's forecast hasn't been uploaded yet.")
                
        return df
        
    except Exception as e:
        st.error(f"Error fetching data from table '{table_name}': {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_available_date_published_dates(table_name: str):
    """Get all available date_published dates for a table"""
    engine = get_db_engine()
    try:
        query = f"""
            SELECT DISTINCT DATE(date_published) as pub_date 
            FROM `{table_name}` 
            ORDER BY pub_date DESC 
            LIMIT 10
        """
        df = pd.read_sql_query(query, engine)
        if not df.empty:
            return [pd.to_datetime(date).date() for date in df['pub_date']]
        return []
    except Exception as e:
        st.error(f"Error getting available dates for '{table_name}': {e}")
        return []

def get_generation_tables():
    """Get all hourly generation tables"""
    engine = get_db_engine()
    inspector = inspect(engine)
    all_table_names = inspector.get_table_names()
    
    generation_tables = [
        t for t in all_table_names 
        if t.endswith('_hourly_generation')
    ]
    return sorted(generation_tables)

def parse_table_name(table_name):
    """Extract region name from table name"""
    return table_name.replace('_hourly_generation', '').replace('_', ' ').title()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Texas Generation", page_icon="⚡", layout="wide")

# --- PAGE CONTENT ---
st.title("⚡ EIA Generation Analysis")
st.markdown("---")

# Get today's date for display
today = date.today()

st.markdown(f"""
**Analyze hourly electricity generation data by source for all regions.**

**📅 Data Filter:** This page shows **ONLY today's forecast data** (date_published = {today})

This ensures you're viewing the most recent published forecast data including load demand, 
natural gas, wind, solar, coal, nuclear, hydro, and other sources across multiple regions.
""")

# Establish DB connection
engine = get_db_engine()
if not engine: 
    st.stop()

# Get available generation tables
generation_tables = get_generation_tables()

if not generation_tables:
    st.warning("⚠️ No hourly generation tables found in the database.")
    st.stop()

# Sidebar controls - FIXED VERSION
with st.sidebar:
    st.title("🧭 Navigation")
    
    # Navigation buttons
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("streamlit_app.py")
    if st.button("📊 Historical OI", use_container_width=True):
        st.switch_page("pages/1_Historical_OI.py")   
#
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
    st.info("📍 **Current Page:** EIA Generation")
    
    # Show today's date prominently
    st.success(f"📅 **Data Filter:** {today}")
    st.caption("Only showing today's published forecast data")
    
    st.subheader("📊 Analysis Controls")
    
    # Table selection
    selected_table = st.selectbox(
        "Select Region:",
        options=generation_tables,
        format_func=parse_table_name,
        help="Choose a region to analyze"
    )
    
    # Show available dates for this table
    if selected_table:
        available_dates = get_available_date_published_dates(selected_table)
        if available_dates:
            st.write("**Available forecast dates:**")
            for avail_date in available_dates[:5]:  # Show last 5 dates
                if avail_date == today:
                    st.write(f"✅ {avail_date} (Today)")
                else:
                    st.write(f"📅 {avail_date}")
        else:
            st.warning("No data available for this table")
    
    # Date range selection for timestamp filtering
    st.subheader("📅 Time Range Filter")
    st.caption("Filter the forecast data by timestamp (forecast periods)")
    
    # Get sample data to determine available timestamp range
    if selected_table:
        sample_data = get_generation_data(selected_table)
        if not sample_data.empty:
            min_date = sample_data['timestamp'].min().date()
            max_date = sample_data['timestamp'].max().date()
            
            st.write(f"Forecast period: {min_date} to {max_date}")
            
            # Default to show next 3 days of forecast
            default_start = min_date
            default_end = min(max_date, min_date + timedelta(days=3))
            
            start_date = st.date_input(
                "Start Date (Forecast)",
                value=default_start,
                min_value=min_date,
                max_value=max_date,
                help="Start of forecast period to display"
            )
            
            end_date = st.date_input(
                "End Date (Forecast)", 
                value=default_end,
                min_value=min_date,
                max_value=max_date,
                help="End of forecast period to display"
            )
            
            if start_date > end_date:
                st.error("Start date must be before end date")
                st.stop()
        else:
            st.error(f"No data available for today ({today}) in selected table")
            start_date = today
            end_date = today
    
    # Chart options
    st.subheader("📈 Chart Options")
    show_load = st.checkbox("Show Load Demand", value=True)
    show_generation = st.checkbox("Show Generation Sources", value=True)
    stack_chart = st.checkbox("Stack Generation Sources", value=False)

# Main content area
if selected_table:
    region_name = parse_table_name(selected_table)
    st.subheader(f"📈 {region_name} Generation Data - {today} Forecast")
    
    # Load data for selected date range (with today's date_published filter)
    df = get_generation_data(selected_table, start_date, end_date)
    
    if df.empty:
        st.warning(f"No forecast data found for {region_name} published on {today} for the selected time range.")
        
        # Show what dates are available
        available_dates = get_available_date_published_dates(selected_table)
        if available_dates:
            st.info(f"💡 Available forecast dates for {region_name}: {', '.join([str(d) for d in available_dates[:5]])}")
        else:
            st.error(f"No forecast data found at all for {region_name}")
    else:
        # Display data info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Points", f"{len(df):,}")
        with col2:
            st.metric("Forecast Days", f"{(end_date - start_date).days + 1}")
        with col3:
            if 'timestamp' in df.columns:
                hours_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
                st.metric("Hours Span", f"{hours_span:.0f}")
        with col4:
            # Show the date_published for verification
            if 'date_published' in df.columns:
                pub_date = pd.to_datetime(df['date_published']).iloc[0].date()
                st.metric("Published", str(pub_date))
        
        # Show forecast info
        forecast_start = df['timestamp'].min().strftime('%Y-%m-%d %H:%M')
        forecast_end = df['timestamp'].max().strftime('%Y-%m-%d %H:%M')
        st.info(f"📊 Showing forecast from {forecast_start} to {forecast_end} (published on {today})")
        
        # Identify columns
        load_col = [col for col in df.columns if 'LOAD' in col.upper()]
        generation_cols = [col for col in df.columns if '_MW' in col and 'LOAD' not in col.upper()]
        
        # Create plots
        if show_load and show_generation:
            # Dual-axis plot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Load Demand', 'Generation by Source'],
                vertical_spacing=0.1,
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            # Load plot
            if load_col:
                load_data = df[load_col[0]].dropna()
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'], 
                        y=load_data,
                        mode='lines',
                        name='Load Demand',
                        line=dict(color='red', width=2),
                        hovertemplate="Time: %{x}<br>Load: %{y:,.0f} MW<extra></extra>"
                    ),
                    row=1, col=1
                )
            
            # Generation plot
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            
            if stack_chart:
                # Stacked area chart
                y_stack = None
                for i, col in enumerate(generation_cols):
                    col_data = df[col].fillna(0)
                    if y_stack is None:
                        y_stack = col_data
                        stackgroup = 'one'
                    else:
                        stackgroup = 'one'
                    
                    source_name = col.replace(f"{region_name.upper()}_", "").replace("_MW", "")
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=col_data,
                            mode='lines',
                            stackgroup=stackgroup,
                            name=source_name,
                            line=dict(color=colors[i % len(colors)]),
                            hovertemplate=f"{source_name}: %{{y:,.0f}} MW<extra></extra>"
                        ),
                        row=2, col=1
                    )
            else:
                # Individual lines
                for i, col in enumerate(generation_cols):
                    col_data = df[col].dropna()
                    source_name = col.replace(f"{region_name.upper()}_", "").replace("_MW", "")
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'], 
                            y=col_data,
                            mode='lines',
                            name=source_name,
                            line=dict(color=colors[i % len(colors)]),
                            hovertemplate=f"{source_name}: %{{y:,.0f}} MW<extra></extra>"
                        ),
                        row=2, col=1
                    )
            
        elif show_load:
            # Load only
            fig = go.Figure()
            if load_col:
                load_data = df[load_col[0]].dropna()
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'], 
                        y=load_data,
                        mode='lines',
                        name='Load Demand',
                        line=dict(color='red', width=2),
                        hovertemplate="Time: %{x}<br>Load: %{y:,.0f} MW<extra></extra>"
                    )
                )
            fig.update_layout(title=f"{region_name} Load Demand - {today} Forecast")
            
        elif show_generation:
            # Generation only
            fig = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            
            for i, col in enumerate(generation_cols):
                col_data = df[col].dropna()
                source_name = col.replace(f"{region_name.upper()}_", "").replace("_MW", "")
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'], 
                        y=col_data,
                        mode='lines',
                        name=source_name,
                        line=dict(color=colors[i % len(colors)]),
                        hovertemplate=f"{source_name}: %{{y:,.0f}} MW<extra></extra>"
                    )
                )
            fig.update_layout(title=f"{region_name} Generation by Source - {today} Forecast")
        
        # Update layout
        fig.update_layout(
            height=600 if show_load and show_generation else 400,
            hovermode='x unified',
            showlegend=True,
            xaxis_title="Time",
            yaxis_title="Power (MW)"
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        # Calculate stats
        stats_data = []
        
        if load_col and not df[load_col[0]].isna().all():
            load_stats = {
                'Source': 'Load Demand',
                'Average (MW)': f"{df[load_col[0]].mean():,.0f}",
                'Peak (MW)': f"{df[load_col[0]].max():,.0f}",
                'Minimum (MW)': f"{df[load_col[0]].min():,.0f}",
                'Total (MWh)': f"{df[load_col[0]].sum():,.0f}"
            }
            stats_data.append(load_stats)
        
        for col in generation_cols:
            if not df[col].isna().all():
                source_name = col.replace(f"{region_name.upper()}_", "").replace("_MW", "")
                gen_stats = {
                    'Source': source_name,
                    'Average (MW)': f"{df[col].mean():,.0f}",
                    'Peak (MW)': f"{df[col].max():,.0f}",
                    'Minimum (MW)': f"{df[col].min():,.0f}",
                    'Total (MWh)': f"{df[col].sum():,.0f}"
                }
                stats_data.append(gen_stats)
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        
        # Raw data sample
        with st.expander("📋 Raw Data Sample"):
            # Show timestamp, date_published, and a few key columns
            display_cols = ['timestamp', 'date_published'] + load_col + generation_cols[:3]
            display_cols = [col for col in display_cols if col in df.columns]
            st.dataframe(df[display_cols].head(24), use_container_width=True)  # Show first 24 hours
            
            if st.button("📥 Download Full Dataset"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{region_name}_generation_forecast_{today}_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )

st.markdown("---")
st.markdown(f"**💾 Data Source:** EIA via automated ingestion script | **🔄 Data Updates:** Every hour | **📅 Current Filter:** {today} forecast data only")