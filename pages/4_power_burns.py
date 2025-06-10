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
        # Base query
        base_query = "SELECT report_date, L48_Power_Burns, date_published FROM power_burns_daily ORDER BY report_date ASC"
        
        # Add date filtering if provided
        if start_date and end_date:
            query = f"""
                SELECT report_date, L48_Power_Burns, date_published 
                FROM power_burns_daily 
                WHERE report_date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY report_date ASC
            """
        else:
            query = base_query
        
        df = pd.read_sql_query(query, engine)
        
        if not df.empty:
            df['report_date'] = pd.to_datetime(df['report_date']).dt.date
            df['date_published'] = pd.to_datetime(df['date_published'])
            
            # FIXED: Divide by 1000 to correct the values
            df['L48_Power_Burns'] = pd.to_numeric(df['L48_Power_Burns'], errors='coerce') / 1000
            
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
    try:
        query = "SELECT MIN(report_date) as min_date, MAX(report_date) as max_date FROM power_burns_daily"
        result = pd.read_sql_query(query, engine)
        if not result.empty:
            min_date = pd.to_datetime(result['min_date'].iloc[0]).date()
            max_date = pd.to_datetime(result['max_date'].iloc[0]).date()
            return min_date, max_date
    except Exception as e:
        st.error(f"Error getting date range: {e}")
    return None, None

def calculate_seasonal_stats(df):
    """Calculate seasonal statistics for power burns"""
    if df.empty:
        return pd.DataFrame()
    
    # Group by month for seasonal patterns
    monthly_stats = df.groupby('month').agg({
        'L48_Power_Burns': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    
    monthly_stats.columns = ['Avg_Burns', 'Std_Dev', 'Min_Burns', 'Max_Burns', 'Data_Points']
    monthly_stats['Month_Name'] = [calendar.month_name[i] for i in monthly_stats.index]
    
    return monthly_stats.reset_index()

def calculate_yearly_stats(df):
    """Calculate yearly statistics for power burns"""
    if df.empty:
        return pd.DataFrame()
    
    yearly_stats = df.groupby('year').agg({
        'L48_Power_Burns': ['mean', 'sum', 'std', 'min', 'max', 'count']
    }).round(2)
    
    yearly_stats.columns = ['Avg_Daily_Burns', 'Total_Annual_Burns', 'Std_Dev', 'Min_Burns', 'Max_Burns', 'Data_Points']
    
    return yearly_stats.reset_index()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Power Burns Analysis", page_icon="🔥", layout="wide")

# --- PAGE CONTENT ---
st.title("🔥 Natural Gas Power Burns Analysis")
st.markdown("---")

st.markdown("""
**Analyze daily natural gas consumption for power generation in the Lower 48 states.**

This page provides comprehensive analysis of power burns data spanning 2019-2028, including seasonal patterns, 
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
    
    st.markdown("---")
    st.info("📍 **Current Page:** Power Burns")
    
    st.subheader("📊 Analysis Controls")
    
    # Get available date range
    min_date, max_date = get_data_date_range()
    
    if min_date and max_date:
        st.write(f"**Available Data:** {min_date} to {max_date}")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Time Series View", "Seasonal Analysis", "Yearly Comparison", "Historical vs Forecast"],
            help="Choose the type of analysis to perform"
        )
        
        # Date range selection based on analysis type
        if analysis_type == "Time Series View":
            # FIXED: Default to 4 days before today and 20 days into future
            today = datetime.now().date()
            default_start = max(min_date, today - timedelta(days=4))
            default_end = min(max_date, today + timedelta(days=20))
            
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                min_value=min_date,
                max_value=max_date
            )
            
            end_date = st.date_input(
                "End Date", 
                value=default_end,  # CHANGED: Now uses default_end instead of max_date
                min_value=min_date,
                max_value=max_date
            )
            
            if start_date > end_date:
                st.error("Start date must be before end date")
                st.stop()
                
        elif analysis_type == "Yearly Comparison":
            # Year selection for comparison
            available_years = list(range(min_date.year, max_date.year + 1))
            selected_years = st.multiselect(
                "Select Years to Compare:",
                options=available_years,
                default=available_years[-3:] if len(available_years) >= 3 else available_years,
                help="Choose years to overlay for comparison"
            )
            
        elif analysis_type == "Historical vs Forecast":
            # Split point for historical vs forecast
            current_year = datetime.now().year
            historical_cutoff = st.date_input(
                "Historical/Forecast Split:",
                value=date(current_year, 1, 1),
                min_value=min_date,
                max_value=max_date,
                help="Data before this date is historical, after is forecast"
            )
        
        # Chart options
        st.subheader("📈 Chart Options")
        show_trend = st.checkbox("Show Trend Line", value=True)
        show_moving_avg = st.checkbox("Show Moving Average", value=False)
        if show_moving_avg:
            ma_days = st.slider("Moving Average Days:", 7, 365, 30)
    else:
        st.error("Unable to determine data date range")
        st.stop()

# Establish DB connection
engine = get_db_engine()
if not engine: 
    st.stop()

# Main content area
if min_date and max_date:
    
    if analysis_type == "Time Series View":
        st.subheader(f"📈 Power Burns Time Series ({start_date} to {end_date})")
        
        # Load data for selected date range
        df = get_power_burns_data(start_date, end_date)
        
        if df.empty:
            st.warning(f"No data found for the selected date range.")
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
                date_span = (end_date - start_date).days
                st.metric("Date Range", f"{date_span} days")
            
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
                title="Daily Natural Gas Power Burns - Lower 48 States",
                xaxis_title="Date",
                yaxis_title="Power Burns (Bcf/d)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Seasonal Analysis":
        st.subheader("🌿 Seasonal Power Burns Analysis")
        
        # Load all data for seasonal analysis
        df = get_power_burns_data()
        
        if df.empty:
            st.warning("No data available for seasonal analysis.")
        else:
            # Calculate seasonal statistics
            seasonal_stats = calculate_seasonal_stats(df)
            
            # Create seasonal plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Average Monthly Power Burns',
                    'Monthly Variation (Min/Max)',
                    'Monthly Standard Deviation',
                    'Data Points by Month'
                ],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Average monthly burns
            fig.add_trace(
                go.Bar(x=seasonal_stats['Month_Name'], y=seasonal_stats['Avg_Burns'],
                       name='Avg Burns', marker_color='#ff6b35'),
                row=1, col=1
            )
            
            # Min/Max range
            fig.add_trace(
                go.Scatter(x=seasonal_stats['Month_Name'], y=seasonal_stats['Max_Burns'],
                          mode='lines+markers', name='Max', line=dict(color='red')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=seasonal_stats['Month_Name'], y=seasonal_stats['Min_Burns'],
                          mode='lines+markers', name='Min', line=dict(color='blue')),
                row=1, col=2
            )
            
            # Standard deviation
            fig.add_trace(
                go.Bar(x=seasonal_stats['Month_Name'], y=seasonal_stats['Std_Dev'],
                       name='Std Dev', marker_color='green'),
                row=2, col=1
            )
            
            # Data points
            fig.add_trace(
                go.Bar(x=seasonal_stats['Month_Name'], y=seasonal_stats['Data_Points'],
                       name='Data Points', marker_color='purple'),
                row=2, col=2
            )
            
            fig.update_layout(height=700, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display seasonal statistics table
            st.subheader("📊 Seasonal Statistics Summary")
            st.dataframe(seasonal_stats, use_container_width=True)
    
    elif analysis_type == "Yearly Comparison" and selected_years:
        st.subheader(f"📊 Yearly Comparison ({', '.join(map(str, selected_years))})")
        
        # Load all data
        df = get_power_burns_data()
        
        if df.empty:
            st.warning("No data available for yearly comparison.")
        else:
            # Filter for selected years
            df_filtered = df[df['year'].isin(selected_years)]
            
            # Create yearly overlay plot
            fig = go.Figure()
            
            colors = ['#ff6b35', '#004e89', '#009639', '#ffa400', '#9b5de5', '#f72585']
            
            for i, year in enumerate(selected_years):
                year_data = df_filtered[df_filtered['year'] == year].copy()
                if not year_data.empty:
                    year_data = year_data.sort_values('day_of_year')
                    
                    fig.add_trace(go.Scatter(
                        x=year_data['day_of_year'],
                        y=year_data['L48_Power_Burns'],
                        mode='lines',
                        name=str(year),
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate=f"{year}<br>Day of Year: %{{x}}<br>Power Burns: %{{y:.1f}} Bcf/d<extra></extra>"
                    ))
            
            fig.update_layout(
                title="Power Burns by Day of Year - Multi-Year Comparison",
                xaxis_title="Day of Year",
                yaxis_title="Power Burns (Bcf/d)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Yearly statistics
            yearly_stats = calculate_yearly_stats(df_filtered)
            st.subheader("📊 Yearly Statistics")
            st.dataframe(yearly_stats, use_container_width=True)
    
    elif analysis_type == "Historical vs Forecast":
        st.subheader(f"🔮 Historical vs Forecast Analysis (Split: {historical_cutoff})")
        
        # Load all data
        df = get_power_burns_data()
        
        if df.empty:
            st.warning("No data available for historical vs forecast analysis.")
        else:
            # Split data into historical and forecast
            df['report_date_dt'] = pd.to_datetime(df['report_date'])
            cutoff_dt = pd.to_datetime(historical_cutoff)
            
            historical_data = df[df['report_date_dt'] < cutoff_dt]
            forecast_data = df[df['report_date_dt'] >= cutoff_dt]
            
            # Create comparison plot
            fig = go.Figure()
            
            if not historical_data.empty:
                fig.add_trace(go.Scatter(
                    x=historical_data['report_date_dt'],
                    y=historical_data['L48_Power_Burns'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='#004e89', width=2),
                    hovertemplate="Historical<br>Date: %{x}<br>Power Burns: %{y:.1f} Bcf/d<extra></extra>"
                ))
            
            if not forecast_data.empty:
                fig.add_trace(go.Scatter(
                    x=forecast_data['report_date_dt'],
                    y=forecast_data['L48_Power_Burns'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#ff6b35', width=2, dash='dash'),
                    hovertemplate="Forecast<br>Date: %{x}<br>Power Burns: %{y:.1f} Bcf/d<extra></extra>"
                ))
            
            # Add vertical line at split point
            fig.add_vline(
                x=cutoff_dt,
                line_dash="dot",
                line_color="red",
                annotation_text="Historical/Forecast Split"
            )
            
            fig.update_layout(
                title="Historical vs Forecast Power Burns",
                xaxis_title="Date",
                yaxis_title="Power Burns (Bcf/d)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                if not historical_data.empty:
                    st.subheader("📊 Historical Statistics")
                    hist_stats = {
                        'Data Points': len(historical_data),
                        'Average Burns': f"{historical_data['L48_Power_Burns'].mean():.1f} Bcf/d",
                        'Peak Burns': f"{historical_data['L48_Power_Burns'].max():.1f} Bcf/d",
                        'Min Burns': f"{historical_data['L48_Power_Burns'].min():.1f} Bcf/d",
                        'Std Deviation': f"{historical_data['L48_Power_Burns'].std():.1f} Bcf/d"
                    }
                    for key, value in hist_stats.items():
                        st.metric(key, value)
            
            with col2:
                if not forecast_data.empty:
                    st.subheader("🔮 Forecast Statistics")
                    forecast_stats = {
                        'Data Points': len(forecast_data),
                        'Average Burns': f"{forecast_data['L48_Power_Burns'].mean():.1f} Bcf/d",
                        'Peak Burns': f"{forecast_data['L48_Power_Burns'].max():.1f} Bcf/d",
                        'Min Burns': f"{forecast_data['L48_Power_Burns'].min():.1f} Bcf/d",
                        'Std Deviation': f"{forecast_data['L48_Power_Burns'].std():.1f} Bcf/d"
                    }
                    for key, value in forecast_stats.items():
                        st.metric(key, value)

    # Raw data section
    with st.expander("📋 Raw Data Sample"):
        sample_df = get_power_burns_data()
        if not sample_df.empty:
            st.dataframe(sample_df.head(50), use_container_width=True)
            
            if st.button("📥 Download Full Dataset"):
                csv = sample_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"power_burns_daily_{min_date}_{max_date}.csv",
                    mime="text/csv"
                )

st.markdown("---")
st.markdown("**💾 Data Source:** EIA Power Burns Data | **🔄 Data Updates:** Daily")