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

@st.cache_data(ttl=3600)
def get_aggregated_generation_data(start_date=None, end_date=None):
    """Fetch and aggregate generation data from all regions"""
    engine = get_db_engine()
    generation_tables = get_generation_tables()
    
    if not generation_tables:
        return pd.DataFrame()
    
    all_data = []
    
    for table in generation_tables:
        try:
            # Get region name from table
            region = table.replace('_hourly_generation', '').upper()
            
            # Build query with date filtering if provided
            if start_date and end_date:
                query = f"""
                    SELECT timestamp, *
                    FROM `{table}` 
                    WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
                    ORDER BY timestamp ASC
                """
            else:
                query = f"SELECT timestamp, * FROM `{table}` ORDER BY timestamp ASC"
            
            df = pd.read_sql_query(query, engine)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Identify load and generation columns
                load_cols = [col for col in df.columns if 'LOAD' in col.upper()]
                generation_cols = [col for col in df.columns if '_MW' in col and 'LOAD' not in col.upper()]
                
                # Standardize column names by removing region prefix
                standardized_df = df[['timestamp']].copy()
                
                # Add load column (standardized name)
                if load_cols:
                    standardized_df['LOAD_MW'] = pd.to_numeric(df[load_cols[0]], errors='coerce')
                
                # Add generation columns with standardized names
                for col in generation_cols:
                    # Extract generation type (NG, WND, SUN, etc.)
                    parts = col.split('_')
                    if len(parts) >= 2:
                        gen_type = parts[-2]  # Second to last part should be the type
                        standardized_name = f"{gen_type}_MW"
                        standardized_df[standardized_name] = pd.to_numeric(df[col], errors='coerce')
                
                # Add region identifier
                standardized_df['region'] = region
                all_data.append(standardized_df)
                
        except Exception as e:
            st.warning(f"Error processing table {table}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all regional data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Group by timestamp and sum across all regions
    numeric_cols = [col for col in combined_df.columns if col.endswith('_MW')]
    
    aggregated_df = combined_df.groupby('timestamp')[numeric_cols].sum().reset_index()
    
    # Remove any columns that are all NaN or zero
    for col in numeric_cols:
        if aggregated_df[col].isna().all() or (aggregated_df[col] == 0).all():
            aggregated_df = aggregated_df.drop(columns=[col])
    
    return aggregated_df

def calculate_day_over_day_changes(df):
    """Calculate day-over-day changes for daily aggregated data"""
    if df.empty:
        return pd.DataFrame()
    
    # Convert to daily averages first
    df['date'] = df['timestamp'].dt.date
    numeric_cols = [col for col in df.columns if col.endswith('_MW')]
    
    daily_df = df.groupby('date')[numeric_cols].mean().reset_index()
    
    # Calculate day-over-day changes
    changes_df = daily_df.copy()
    
    for col in numeric_cols:
        # Calculate absolute change
        changes_df[f"{col}_change"] = daily_df[col].diff()
        # Calculate percentage change
        changes_df[f"{col}_pct_change"] = daily_df[col].pct_change() * 100
    
    return changes_df

@st.cache_data(ttl=3600)
def get_data_date_range():
    """Get the available date range across all generation tables"""
    engine = get_db_engine()
    generation_tables = get_generation_tables()
    
    if not generation_tables:
        return None, None
    
    min_dates = []
    max_dates = []
    
    for table in generation_tables:
        try:
            query = f"SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM `{table}`"
            result = pd.read_sql_query(query, engine)
            if not result.empty and result['min_date'].iloc[0] is not None:
                min_dates.append(pd.to_datetime(result['min_date'].iloc[0]))
                max_dates.append(pd.to_datetime(result['max_date'].iloc[0]))
        except Exception:
            continue
    
    if min_dates and max_dates:
        return min(min_dates).date(), max(max_dates).date()
    
    return None, None

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Net Changes", page_icon="📈", layout="wide")

# --- PAGE CONTENT ---
st.title("📈 Net Changes Analysis")
st.markdown("---")

st.markdown("""
**Analyze day-over-day changes in total electricity generation across all regions.**

This page aggregates hourly generation data from all available regions (California, Texas, Florida, etc.) 
and compares forecast changes day-over-day for each generation source.
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
    
    st.markdown("---")
    st.info("📍 **Current Page:** Net Changes")
    
    st.subheader("📊 Analysis Controls")
    
    # Get available date range
    min_date, max_date = get_data_date_range()
    
    if min_date and max_date:
        st.write(f"**Available Data:** {min_date} to {max_date}")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Day-over-Day Changes", "Cumulative Trends", "Generation Mix Evolution"],
            help="Choose the type of analysis to perform"
        )
        
        # Date range selection
        st.subheader("📅 Date Range")
        
        # Default to last 30 days for performance
        default_start = max(min_date, max_date - timedelta(days=30))
        
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
        
        # Chart options
        st.subheader("📈 Chart Options")
        show_percentage = st.checkbox("Show Percentage Changes", value=True)
        show_absolute = st.checkbox("Show Absolute Changes", value=False)
        filter_small_changes = st.checkbox("Filter Small Changes", value=True)
        if filter_small_changes:
            min_change_threshold = st.slider("Minimum Change (MW):", 10, 1000, 100)
    else:
        st.error("Unable to determine data date range")
        st.stop()

# Establish DB connection
engine = get_db_engine()
if not engine: 
    st.stop()

# Main content area
if min_date and max_date:
    
    # Load and process data
    with st.spinner("Loading and aggregating data from all regions..."):
        df = get_aggregated_generation_data(start_date, end_date)
    
    if df.empty:
        st.warning(f"No data found for the selected date range ({start_date} to {end_date}).")
    else:
        # Show available regions and generation types
        generation_tables = get_generation_tables()
        regions = [t.replace('_hourly_generation', '').replace('_', ' ').title() for t in generation_tables]
        generation_types = [col.replace('_MW', '') for col in df.columns if col.endswith('_MW')]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Regions Aggregated", len(regions))
        with col2:
            st.metric("Generation Types", len(generation_types))
        with col3:
            st.metric("Data Points", f"{len(df):,}")
        
        st.info(f"**Regions:** {', '.join(regions)}")
        st.info(f"**Generation Types:** {', '.join(generation_types)}")
        
        if analysis_type == "Day-over-Day Changes":
            st.subheader("📊 Day-over-Day Generation Changes")
            
            # Calculate day-over-day changes
            changes_df = calculate_day_over_day_changes(df)
            
            if changes_df.empty:
                st.warning("Unable to calculate day-over-day changes.")
            else:
                # Filter out the first row (no previous day to compare)
                changes_df = changes_df.iloc[1:].copy()
                
                # Get change columns
                change_cols = [col for col in changes_df.columns if col.endswith('_change')]
                pct_change_cols = [col for col in changes_df.columns if col.endswith('_pct_change')]
                
                # Create visualization
                if show_percentage and pct_change_cols:
                    fig = go.Figure()
                    
                    colors = ['#ff6b35', '#004e89', '#009639', '#ffa400', '#9b5de5', '#f72585', '#00b4d8', '#90e0ef']
                    
                    for i, col in enumerate(pct_change_cols):
                        gen_type = col.replace('_MW_pct_change', '')
                        
                        # Filter small changes if requested
                        plot_data = changes_df.copy()
                        if filter_small_changes:
                            abs_change_col = f"{gen_type}_MW_change"
                            if abs_change_col in changes_df.columns:
                                plot_data = plot_data[abs(plot_data[abs_change_col]) >= min_change_threshold]
                        
                        if not plot_data.empty:
                            fig.add_trace(go.Scatter(
                                x=plot_data['date'],
                                y=plot_data[col],
                                mode='lines+markers',
                                name=gen_type,
                                line=dict(color=colors[i % len(colors)], width=2),
                                hovertemplate=f"{gen_type}<br>Date: %{{x}}<br>Change: %{{y:.2f}}%<extra></extra>"
                            ))
                    
                    fig.update_layout(
                        title="Day-over-Day Percentage Changes in Generation by Source",
                        xaxis_title="Date",
                        yaxis_title="Percentage Change (%)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    # Add horizontal line at 0
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                if show_absolute and change_cols:
                    fig2 = go.Figure()
                    
                    colors = ['#ff6b35', '#004e89', '#009639', '#ffa400', '#9b5de5', '#f72585', '#00b4d8', '#90e0ef']
                    
                    for i, col in enumerate(change_cols):
                        gen_type = col.replace('_MW_change', '')
                        
                        # Filter small changes if requested
                        plot_data = changes_df.copy()
                        if filter_small_changes:
                            plot_data = plot_data[abs(plot_data[col]) >= min_change_threshold]
                        
                        if not plot_data.empty:
                            fig2.add_trace(go.Scatter(
                                x=plot_data['date'],
                                y=plot_data[col],
                                mode='lines+markers',
                                name=gen_type,
                                line=dict(color=colors[i % len(colors)], width=2),
                                hovertemplate=f"{gen_type}<br>Date: %{{x}}<br>Change: %{{y:,.0f}} MW<extra></extra>"
                            ))
                    
                    fig2.update_layout(
                        title="Day-over-Day Absolute Changes in Generation by Source",
                        xaxis_title="Date",
                        yaxis_title="Change (MW)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    # Add horizontal line at 0
                    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Summary statistics
                st.subheader("📊 Change Statistics Summary")
                
                if pct_change_cols:
                    stats_data = []
                    
                    for col in pct_change_cols:
                        gen_type = col.replace('_MW_pct_change', '')
                        valid_data = changes_df[col].dropna()
                        
                        if not valid_data.empty:
                            stats_data.append({
                                'Generation Type': gen_type,
                                'Avg Daily Change (%)': f"{valid_data.mean():.2f}%",
                                'Max Increase (%)': f"{valid_data.max():.2f}%",
                                'Max Decrease (%)': f"{valid_data.min():.2f}%",
                                'Volatility (Std Dev)': f"{valid_data.std():.2f}%",
                                'Days with Data': len(valid_data)
                            })
                    
                    if stats_data:
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
        
        elif analysis_type == "Cumulative Trends":
            st.subheader("📈 Cumulative Generation Trends")
            
            # Convert to daily averages
            df['date'] = df['timestamp'].dt.date
            numeric_cols = [col for col in df.columns if col.endswith('_MW')]
            daily_df = df.groupby('date')[numeric_cols].mean().reset_index()
            
            # Create cumulative trends plot
            fig = go.Figure()
            colors = ['#ff6b35', '#004e89', '#009639', '#ffa400', '#9b5de5', '#f72585', '#00b4d8', '#90e0ef']
            
            for i, col in enumerate(numeric_cols):
                gen_type = col.replace('_MW', '')
                
                fig.add_trace(go.Scatter(
                    x=daily_df['date'],
                    y=daily_df[col],
                    mode='lines',
                    name=gen_type,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f"{gen_type}<br>Date: %{{x}}<br>Generation: %{{y:,.0f}} MW<extra></extra>"
                ))
            
            fig.update_layout(
                title="Daily Average Generation by Source (All Regions Combined)",
                xaxis_title="Date",
                yaxis_title="Generation (MW)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Generation Mix Evolution":
            st.subheader("🥧 Generation Mix Evolution")
            
            # Convert to daily totals and calculate percentages
            df['date'] = df['timestamp'].dt.date
            numeric_cols = [col for col in df.columns if col.endswith('_MW') and col != 'LOAD_MW']
            daily_df = df.groupby('date')[numeric_cols].mean().reset_index()
            
            # Calculate total generation and percentages
            daily_df['total_generation'] = daily_df[numeric_cols].sum(axis=1)
            
            for col in numeric_cols:
                daily_df[f"{col}_pct"] = (daily_df[col] / daily_df['total_generation']) * 100
            
            # Create stacked area chart
            fig = go.Figure()
            colors = ['#ff6b35', '#004e89', '#009639', '#ffa400', '#9b5de5', '#f72585', '#00b4d8', '#90e0ef']
            
            for i, col in enumerate(numeric_cols):
                gen_type = col.replace('_MW', '')
                pct_col = f"{col}_pct"
                
                fig.add_trace(go.Scatter(
                    x=daily_df['date'],
                    y=daily_df[pct_col],
                    mode='lines',
                    stackgroup='one',
                    name=gen_type,
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate=f"{gen_type}<br>Date: %{{x}}<br>Share: %{{y:.1f}}%<extra></extra>"
                ))
            
            fig.update_layout(
                title="Generation Mix Evolution (Percentage of Total)",
                xaxis_title="Date",
                yaxis_title="Percentage of Total Generation (%)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

        # Raw data section
        with st.expander("📋 Raw Data Sample"):
            if analysis_type == "Day-over-Day Changes" and not changes_df.empty:
                st.dataframe(changes_df.head(20), use_container_width=True)
            else:
                # Show daily averages
                df['date'] = df['timestamp'].dt.date
                numeric_cols = [col for col in df.columns if col.endswith('_MW')]
                daily_sample = df.groupby('date')[numeric_cols].mean().reset_index().head(20)
                st.dataframe(daily_sample, use_container_width=True)

st.markdown("---")
st.markdown("**💾 Data Source:** EIA Generation Data (All Regions Aggregated) | **🔄 Data Updates:** Hourly")