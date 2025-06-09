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
    """Fetch generation data from the specified table"""
    engine = get_db_engine()
    df = pd.DataFrame()
    
    try:
        # Base query
        query = f"SELECT * FROM `{table_name}` ORDER BY timestamp ASC"
        
        # Add date filtering if provided
        if start_date and end_date:
            query = f"""
                SELECT * FROM `{table_name}` 
                WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY timestamp ASC
            """
        
        df = pd.read_sql_query(query, engine)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Convert numeric columns
            numeric_columns = [col for col in df.columns if col not in ['timestamp', 'date_published']]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
        
    except Exception as e:
        st.error(f"Error fetching data from table '{table_name}': {e}")
        return pd.DataFrame()

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
st.set_page_config(page_title="EIA Generation", page_icon="⚡", layout="wide")

# --- PAGE CONTENT ---
st.title("⚡ EIA Generation Analysis")
st.markdown("---")

st.markdown("""
**Analyze hourly electricity generation data by source for all regions.**

This page provides detailed analysis of electricity generation patterns including load demand, 
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
    
    if st.button("🔥 Power Burns", use_container_width=True):
        st.switch_page("pages/4_Power_Burns.py")
    st.markdown("---")
    st.info("📍 **Current Page:** EIA Generation")
    
    st.subheader("📊 Analysis Controls")
    
    # Table selection
    selected_table = st.selectbox(
        "Select Region:",
        options=generation_tables,
        format_func=parse_table_name,
        help="Choose a region to analyze"
    )
    
    # Date range selection
    st.subheader("📅 Date Range")
    
    # Get min/max dates for the selected table
    sample_data = get_generation_data(selected_table)
    if not sample_data.empty:
        min_date = sample_data['timestamp'].min().date()
        max_date = sample_data['timestamp'].max().date()
        
        st.write(f"Available: {min_date} to {max_date}")
        
        # Default to last 7 days
        default_start = max(min_date, max_date - timedelta(days=7))
        
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.date_input(
            "End Date", 
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
        
        if start_date > end_date:
            st.error("Start date must be before end date")
            st.stop()
    else:
        st.error("No data available for selected table")
        st.stop()
    
    # Chart options
    st.subheader("📈 Chart Options")
    show_load = st.checkbox("Show Load Demand", value=True)
    show_generation = st.checkbox("Show Generation Sources", value=True)
    stack_chart = st.checkbox("Stack Generation Sources", value=False)

# Main content area
if selected_table:
    region_name = parse_table_name(selected_table)
    st.subheader(f"📈 {region_name} Generation Data")
    
    # Load data for selected date range
    df = get_generation_data(selected_table, start_date, end_date)
    
    if df.empty:
        st.warning(f"No data found for {region_name} in the selected date range.")
    else:
        # Display data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", f"{len(df):,}")
        with col2:
            st.metric("Date Range", f"{(end_date - start_date).days} days")
        with col3:
            if 'timestamp' in df.columns:
                hours_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
                st.metric("Hours Span", f"{hours_span:.0f}")
        
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
            fig.update_layout(title=f"{region_name} Load Demand")
            
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
            fig.update_layout(title=f"{region_name} Generation by Source")
        
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
            st.dataframe(df.head(24), use_container_width=True)  # Show first 24 hours
            
            if st.button("📥 Download Full Dataset"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{region_name}_generation_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )

st.markdown("---")
st.markdown("**💾 Data Source:** EIA via automated ingestion script | **🔄 Data Updates:** Every hour")