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
def get_available_forecast_dates():
    """Get available date_published dates across all tables"""
    engine = get_db_engine()
    generation_tables = get_generation_tables()
    
    if not generation_tables:
        return []
    
    all_dates = set()
    
    for table in generation_tables:
        try:
            query = f"""
                SELECT DISTINCT DATE(date_published) as pub_date 
                FROM `{table}` 
                ORDER BY pub_date DESC 
                LIMIT 10
            """
            result = pd.read_sql_query(query, engine)
            if not result.empty:
                dates = [pd.to_datetime(d).date() for d in result['pub_date']]
                all_dates.update(dates)
        except Exception:
            continue
    
    return sorted(list(all_dates), reverse=True)

@st.cache_data(ttl=3600)
def get_aggregated_generation_data_for_date(pub_date, start_timestamp=None, end_timestamp=None):
    """Fetch and aggregate generation data from all regions for a specific publication date"""
    engine = get_db_engine()
    generation_tables = get_generation_tables()
    
    if not generation_tables:
        return pd.DataFrame()
    
    all_data = []
    
    for table in generation_tables:
        try:
            # Get region name from table
            region = table.replace('_hourly_generation', '').upper()
            
            # Build query for specific publication date
            base_query = f"""
                SELECT *
                FROM `{table}` 
                WHERE DATE(date_published) = '{pub_date}'
            """
            
            if start_timestamp and end_timestamp:
                query = f"""
                    {base_query}
                    AND DATE(timestamp) BETWEEN '{start_timestamp}' AND '{end_timestamp}'
                    ORDER BY timestamp ASC
                """
            else:
                query = f"{base_query} ORDER BY timestamp ASC"
            
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
            # st.warning(f"Error processing table {table} for {pub_date}: {e}")
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

def find_common_timestamps(df1, df2):
    """Find common timestamps between two dataframes"""
    if df1.empty or df2.empty:
        return []
    
    timestamps1 = set(df1['timestamp'].dt.floor('H'))  # Round to hour for comparison
    timestamps2 = set(df2['timestamp'].dt.floor('H'))
    
    common_timestamps = timestamps1.intersection(timestamps2)
    return sorted(list(common_timestamps))

def calculate_forecast_changes(df_new, df_old, common_timestamps):
    """Calculate changes between two forecast datasets for common timestamps"""
    if df_new.empty or df_old.empty or not common_timestamps:
        return pd.DataFrame()
    
    # Filter both dataframes to common timestamps (rounded to hour)
    df_new_filtered = df_new[df_new['timestamp'].dt.floor('H').isin(common_timestamps)].copy()
    df_old_filtered = df_old[df_old['timestamp'].dt.floor('H').isin(common_timestamps)].copy()
    
    # Round timestamps for matching
    df_new_filtered['timestamp_hour'] = df_new_filtered['timestamp'].dt.floor('H')
    df_old_filtered['timestamp_hour'] = df_old_filtered['timestamp'].dt.floor('H')
    
    # Merge on rounded timestamp
    merged = pd.merge(
        df_new_filtered, 
        df_old_filtered, 
        on='timestamp_hour', 
        suffixes=('_new', '_old')
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    # Calculate changes for each generation type
    numeric_cols = [col for col in df_new.columns if col.endswith('_MW')]
    changes_df = merged[['timestamp_hour']].copy()
    changes_df.rename(columns={'timestamp_hour': 'timestamp'}, inplace=True)
    
    for col in numeric_cols:
        if f"{col}_new" in merged.columns and f"{col}_old" in merged.columns:
            # Absolute change
            changes_df[f"{col}_change"] = merged[f"{col}_new"] - merged[f"{col}_old"]
            # Percentage change
            changes_df[f"{col}_pct_change"] = (
                (merged[f"{col}_new"] - merged[f"{col}_old"]) / 
                (merged[f"{col}_old"].replace(0, np.nan)) * 100
            )
            # Store both values for reference
            changes_df[f"{col}_new"] = merged[f"{col}_new"]
            changes_df[f"{col}_old"] = merged[f"{col}_old"]
    
    return changes_df

def calculate_daily_forecast_changes(changes_df):
    """Calculate daily aggregated forecast changes"""
    if changes_df.empty:
        return pd.DataFrame()
    
    changes_df['date'] = pd.to_datetime(changes_df['timestamp']).dt.date
    
    # Get change columns
    change_cols = [col for col in changes_df.columns if col.endswith('_change')]
    pct_change_cols = [col for col in changes_df.columns if col.endswith('_pct_change')]
    
    # Aggregate by date
    daily_changes = changes_df.groupby('date').agg({
        **{col: 'mean' for col in change_cols},
        **{col: 'mean' for col in pct_change_cols}
    }).reset_index()
    
    return daily_changes

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Forecast Changes", page_icon="📈", layout="wide")

# --- PAGE CONTENT ---
st.title("📈 Forecast vs Forecast Analysis")
st.markdown("---")

st.markdown("""
**Analyze how electricity generation forecasts change between publication dates.**

This page compares today's forecast against yesterday's (and up to 3 days back) to show how predictions 
for the same time periods evolve as new data becomes available.
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
    if st.button("🔥 Power Burns", use_container_width=True):
        st.switch_page("pages/4_Power_Burns.py")        
    if st.button("📈 Net Changes", use_container_width=True):
        st.switch_page("pages/5_Net_Changes.py")
        
    if st.button("📊 Tape Analysis", use_container_width=True):
        st.switch_page("pages/6_Tape_Analysis.py")
    
    st.markdown("---")
    st.info("📍 **Current Page:** Forecast Changes")
    
    st.subheader("📊 Analysis Controls")
    
    # Get available forecast dates
    available_dates = get_available_forecast_dates()
    
    if not available_dates:
        st.error("No forecast dates found")
        st.stop()
    
    st.write("**Available Forecast Dates:**")
    for i, avail_date in enumerate(available_dates[:5]):
        if i == 0:
            st.write(f"📅 {avail_date} (Latest)")
        else:
            st.write(f"📅 {avail_date}")
    
    # Date selection for comparison
    st.subheader("🔄 Forecast Comparison")
    
    # Default to comparing latest vs previous
    if len(available_dates) >= 2:
        default_new = available_dates[0]  # Latest
        default_old = available_dates[1]  # Previous
    else:
        default_new = available_dates[0] if available_dates else date.today()
        default_old = default_new
    
    new_forecast_date = st.selectbox(
        "📈 Newer Forecast:",
        options=available_dates,
        index=0,
        help="The more recent forecast to compare"
    )
    
    # Filter old forecast options to only dates before the new one
    old_forecast_options = [d for d in available_dates if d < new_forecast_date]
    
    if not old_forecast_options:
        st.error(f"No older forecasts available before {new_forecast_date}")
        st.stop()
    
    old_forecast_date = st.selectbox(
        "📉 Older Forecast:",
        options=old_forecast_options,
        index=0,
        help="The older forecast to compare against"
    )
    
    days_diff = (new_forecast_date - old_forecast_date).days
    st.info(f"📊 Comparing forecasts {days_diff} day(s) apart")
    
    # Analysis type
    analysis_type = st.selectbox(
        "Analysis Type:",
        ["Hourly Changes", "Daily Changes", "Generation Mix Changes"],
        help="Choose how to analyze the forecast differences"
    )
    
    # Chart options
    st.subheader("📈 Chart Options")
    show_percentage = st.checkbox("Show Percentage Changes", value=True)
    show_absolute = st.checkbox("Show Absolute Changes", value=False)
    filter_small_changes = st.checkbox("Filter Small Changes", value=True)
    if filter_small_changes:
        min_change_threshold = st.slider("Minimum Change (MW):", 10, 500, 50)

# Establish DB connection
engine = get_db_engine()
if not engine: 
    st.stop()

# Main content area
st.subheader(f"📊 Forecast Comparison: {new_forecast_date} vs {old_forecast_date}")

# Load data for both forecast dates
with st.spinner("Loading forecast data for comparison..."):
    df_new = get_aggregated_generation_data_for_date(new_forecast_date)
    df_old = get_aggregated_generation_data_for_date(old_forecast_date)

if df_new.empty and df_old.empty:
    st.error("No data found for either forecast date")
elif df_new.empty:
    st.error(f"No data found for newer forecast ({new_forecast_date})")
elif df_old.empty:
    st.error(f"No data found for older forecast ({old_forecast_date})")
else:
    # Find common timestamps
    common_timestamps = find_common_timestamps(df_new, df_old)
    
    if not common_timestamps:
        st.error("No overlapping forecast periods found between the two dates")
    else:
        # Calculate forecast changes
        changes_df = calculate_forecast_changes(df_new, df_old, common_timestamps)
        
        if changes_df.empty:
            st.error("Unable to calculate forecast changes")
        else:
            # Show data summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Common Hours", f"{len(changes_df):,}")
            with col2:
                forecast_start = changes_df['timestamp'].min().strftime('%Y-%m-%d')
                forecast_end = changes_df['timestamp'].max().strftime('%Y-%m-%d')
                st.metric("Forecast Period", f"{forecast_start} to {forecast_end}")
            with col3:
                st.metric("Newer Forecast", str(new_forecast_date))
            with col4:
                st.metric("Older Forecast", str(old_forecast_date))
            
            # Get generation types
            generation_types = [col.replace('_MW_change', '') for col in changes_df.columns if col.endswith('_MW_change')]
            st.info(f"**Generation Types:** {', '.join(generation_types)}")
            
            if analysis_type == "Hourly Changes":
                st.subheader("⏰ Hourly Forecast Changes")
                
                change_cols = [col for col in changes_df.columns if col.endswith('_change')]
                pct_change_cols = [col for col in changes_df.columns if col.endswith('_pct_change')]
                
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
                                x=plot_data['timestamp'],
                                y=plot_data[col],
                                mode='lines+markers',
                                name=gen_type,
                                line=dict(color=colors[i % len(colors)], width=2),
                                hovertemplate=f"{gen_type}<br>Time: %{{x}}<br>Change: %{{y:.2f}}%<extra></extra>"
                            ))
                    
                    fig.update_layout(
                        title=f"Hourly Forecast Changes: {new_forecast_date} vs {old_forecast_date} (%)",
                        xaxis_title="Forecast Time",
                        yaxis_title="Percentage Change (%)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    st.plotly_chart(fig, use_container_width=True)
                
                if show_absolute and change_cols:
                    fig2 = go.Figure()
                    colors = ['#ff6b35', '#004e89', '#009639', '#ffa400', '#9b5de5', '#f72585', '#00b4d8', '#90e0ef']
                    
                    for i, col in enumerate(change_cols):
                        gen_type = col.replace('_MW_change', '')
                        
                        plot_data = changes_df.copy()
                        if filter_small_changes:
                            plot_data = plot_data[abs(plot_data[col]) >= min_change_threshold]
                        
                        if not plot_data.empty:
                            fig2.add_trace(go.Scatter(
                                x=plot_data['timestamp'],
                                y=plot_data[col],
                                mode='lines+markers',
                                name=gen_type,
                                line=dict(color=colors[i % len(colors)], width=2),
                                hovertemplate=f"{gen_type}<br>Time: %{{x}}<br>Change: %{{y:,.0f}} MW<extra></extra>"
                            ))
                    
                    fig2.update_layout(
                        title=f"Hourly Forecast Changes: {new_forecast_date} vs {old_forecast_date} (MW)",
                        xaxis_title="Forecast Time",
                        yaxis_title="Change (MW)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    st.plotly_chart(fig2, use_container_width=True)
            
            elif analysis_type == "Daily Changes":
                st.subheader("📅 Daily Aggregated Forecast Changes")
                
                daily_changes = calculate_daily_forecast_changes(changes_df)
                
                if not daily_changes.empty:
                    change_cols = [col for col in daily_changes.columns if col.endswith('_change')]
                    pct_change_cols = [col for col in daily_changes.columns if col.endswith('_pct_change')]
                    
                    if show_percentage and pct_change_cols:
                        fig = go.Figure()
                        colors = ['#ff6b35', '#004e89', '#009639', '#ffa400', '#9b5de5', '#f72585', '#00b4d8', '#90e0ef']
                        
                        for i, col in enumerate(pct_change_cols):
                            gen_type = col.replace('_MW_pct_change', '')
                            
                            fig.add_trace(go.Bar(
                                x=daily_changes['date'],
                                y=daily_changes[col],
                                name=gen_type,
                                marker_color=colors[i % len(colors)],
                                hovertemplate=f"{gen_type}<br>Date: %{{x}}<br>Avg Change: %{{y:.2f}}%<extra></extra>"
                            ))
                        
                        fig.update_layout(
                            title=f"Daily Average Forecast Changes: {new_forecast_date} vs {old_forecast_date} (%)",
                            xaxis_title="Forecast Date",
                            yaxis_title="Average Daily Change (%)",
                            height=500
                        )
                        
                        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                        st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Generation Mix Changes":
                st.subheader("🥧 Generation Mix Forecast Changes")
                
                # Calculate generation mix changes
                generation_cols = [col for col in changes_df.columns if col.endswith('_MW_new')]
                
                if generation_cols:
                    # Calculate totals for both forecasts
                    changes_df['total_new'] = changes_df[[col for col in changes_df.columns if col.endswith('_MW_new')]].sum(axis=1)
                    changes_df['total_old'] = changes_df[[col for col in changes_df.columns if col.endswith('_MW_old')]].sum(axis=1)
                    
                    # Calculate mix percentages
                    mix_changes = []
                    for col in generation_cols:
                        gen_type = col.replace('_MW_new', '')
                        old_col = f"{gen_type}_MW_old"
                        
                        if old_col in changes_df.columns:
                            # Calculate mix percentage for both forecasts
                            new_mix = (changes_df[col] / changes_df['total_new'] * 100).mean()
                            old_mix = (changes_df[old_col] / changes_df['total_old'] * 100).mean()
                            mix_change = new_mix - old_mix
                            
                            mix_changes.append({
                                'Generation Type': gen_type,
                                'Old Mix (%)': f"{old_mix:.1f}%",
                                'New Mix (%)': f"{new_mix:.1f}%",
                                'Mix Change (%)': f"{mix_change:+.1f}%"
                            })
                    
                    if mix_changes:
                        mix_df = pd.DataFrame(mix_changes)
                        st.dataframe(mix_df, use_container_width=True)
            
            # Summary statistics
            st.subheader("📊 Forecast Change Statistics")
            
            change_cols = [col for col in changes_df.columns if col.endswith('_change')]
            if change_cols:
                stats_data = []
                
                for col in change_cols:
                    gen_type = col.replace('_MW_change', '')
                    valid_data = changes_df[col].dropna()
                    
                    if not valid_data.empty:
                        stats_data.append({
                            'Generation Type': gen_type,
                            'Avg Change (MW)': f"{valid_data.mean():.1f}",
                            'Max Increase (MW)': f"{valid_data.max():.1f}",
                            'Max Decrease (MW)': f"{valid_data.min():.1f}",
                            'Std Dev (MW)': f"{valid_data.std():.1f}",
                            'Hours with Data': len(valid_data)
                        })
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True)
            
            # Raw data section
            with st.expander("📋 Forecast Comparison Data"):
                # Show sample of changes
                display_cols = ['timestamp'] + [col for col in changes_df.columns if '_change' in col or col.endswith('_new') or col.endswith('_old')][:10]
                st.dataframe(changes_df[display_cols].head(24), use_container_width=True)

st.markdown("---")
st.markdown("**💾 Data Source:** EIA Generation Data (All Regions) | **🔄 Analysis:** Forecast Evolution Tracking")