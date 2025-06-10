import streamlit as st

# --- PAGE CONFIGURATION MUST BE FIRST ---
st.set_page_config(page_title="Power Burns Analysis", page_icon="🔥", layout="wide")

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

def calculate_day_over_day_changes(df):
    """Calculate day-over-day changes for power burns data"""
    if df.empty:
        return pd.DataFrame()
    
    # Sort by report_date to ensure proper order
    df_sorted = df.sort_values('report_date').copy()
    
    # Calculate day-over-day changes
    df_sorted['previous_day_burns'] = df_sorted['L48_Power_Burns'].shift(1)
    df_sorted['day_change_absolute'] = df_sorted['L48_Power_Burns'] - df_sorted['previous_day_burns']
    df_sorted['day_change_percentage'] = (df_sorted['day_change_absolute'] / df_sorted['previous_day_burns']) * 100
    
    # Remove the first row (no previous day to compare)
    df_changes = df_sorted.iloc[1:].copy()
    
    # Add rolling statistics
    df_changes['rolling_avg_change'] = df_changes['day_change_absolute'].rolling(window=7, center=True).mean()
    df_changes['rolling_std_change'] = df_changes['day_change_absolute'].rolling(window=7, center=True).std()
    
    return df_changes

def get_forecast_evolution_data(target_start_date, target_end_date):
    """Get how forecasts for specific dates evolved over time"""
    engine = get_db_engine()
    
    try:
        # Get all forecasts for the target date range, grouped by publication date
        query = f"""
            SELECT report_date, L48_Power_Burns, date_published 
            FROM power_burns_daily 
            WHERE report_date BETWEEN '{target_start_date}' AND '{target_end_date}'
            ORDER BY report_date ASC, date_published ASC
        """
        
        df = pd.read_sql_query(query, engine)
        
        if not df.empty:
            df['report_date'] = pd.to_datetime(df['report_date']).dt.date
            df['date_published'] = pd.to_datetime(df['date_published']).dt.date
            
            # Divide by 1000 to correct values
            df['L48_Power_Burns'] = pd.to_numeric(df['L48_Power_Burns'], errors='coerce') / 1000
            
        return df
        
    except Exception as e:
        st.error(f"Error fetching forecast evolution data: {e}")
        return pd.DataFrame()

# --- PAGE CONTENT ---
st.title("🔥 Natural Gas Power Burns Analysis")
st.markdown("---")

st.markdown("""
**Analyze daily natural gas consumption for power generation in the Lower 48 states.**

This page provides comprehensive analysis of power burns data spanning 2019-2028, including seasonal patterns, 
yearly trends, and forecast evolution analysis.
""")

# Sidebar controls
with st.sidebar:
    st.title("🧭 Navigation")
    
    # Navigation buttons - ALL 6 PAGES
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
    
    st.subheader("📊 Analysis Controls")
    
    # Get available date range
    min_date, max_date = get_data_date_range()
    
    if min_date and max_date:
        st.write(f"**Available Data:** {min_date} to {max_date}")
        
        # Analysis type selection - UPDATED
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Time Series View", "Day-over-Day Changes", "Seasonal Analysis", "Yearly Comparison", "Forecast Evolution"],
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
                value=default_end,
                min_value=min_date,
                max_value=max_date
            )
            
            if start_date > end_date:
                st.error("Start date must be before end date")
                st.stop()
                
        elif analysis_type == "Day-over-Day Changes":
            # NEW: Date range selection for day-over-day analysis
            st.subheader("📅 Analysis Date Range")
            
            today = datetime.now().date()
            default_start = max(min_date, today - timedelta(days=30))
            default_end = min(max_date, today + timedelta(days=7))
            
            dod_start_date = st.date_input(
                "Start Date:",
                value=default_start,
                min_value=min_date,
                max_value=max_date,
                help="Starting date for day-over-day analysis"
            )
            
            dod_end_date = st.date_input(
                "End Date:",
                value=default_end,
                min_value=min_date,
                max_value=max_date,
                help="Ending date for day-over-day analysis"
            )
            
            if dod_start_date > dod_end_date:
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
            
        elif analysis_type == "Forecast Evolution":
            # NEW: Date range selection for forecast evolution analysis
            st.subheader("📅 Forecast Date Range")
            
            # Select the dates you want to see forecasts for
            forecast_start_date = st.date_input(
                "Forecast Start Date:",
                value=max(min_date, datetime.now().date() - timedelta(days=7)),
                min_value=min_date,
                max_value=max_date,
                help="Starting date for forecast period you want to analyze"
            )
            
            forecast_end_date = st.date_input(
                "Forecast End Date:",
                value=min(max_date, datetime.now().date() + timedelta(days=14)),
                min_value=min_date,
                max_value=max_date,
                help="Ending date for forecast period you want to analyze"
            )
            
            if forecast_start_date > forecast_end_date:
                st.error("Forecast start date must be before end date")
                st.stop()
        
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
    
    elif analysis_type == "Day-over-Day Changes":
        st.subheader(f"📊 Day-over-Day Power Burns Changes ({dod_start_date} to {dod_end_date})")
        
        # Load data for selected date range
        df = get_power_burns_data(dod_start_date, dod_end_date)
        
        if df.empty:
            st.warning("No data available for day-over-day analysis in the selected date range.")
        else:
            # Calculate changes
            changes_df = calculate_day_over_day_changes(df)
            
            if changes_df.empty:
                st.warning("Insufficient data for day-over-day analysis (need at least 2 days).")
            else:
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                avg_daily_change = changes_df['day_change_absolute'].mean()
                max_increase = changes_df['day_change_absolute'].max()
                max_decrease = changes_df['day_change_absolute'].min()
                volatility = changes_df['day_change_absolute'].std()
                
                with col1:
                    st.metric("Avg Daily Change", f"{avg_daily_change:+.1f} Bcf/d")
                with col2:
                    st.metric("Largest Increase", f"{max_increase:+.1f} Bcf/d")
                with col3:
                    st.metric("Largest Decrease", f"{max_decrease:+.1f} Bcf/d")
                with col4:
                    st.metric("Daily Volatility", f"{volatility:.1f} Bcf/d")
                
                # Create day-over-day change visualization
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=[
                        'Daily Power Burns with Changes',
                        'Day-over-Day Absolute Changes (Bcf/d)', 
                        'Day-over-Day Percentage Changes (%)'
                    ],
                    vertical_spacing=0.08,
                    row_heights=[0.4, 0.3, 0.3]
                )
                
                # Original power burns line
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(changes_df['report_date']),
                        y=changes_df['L48_Power_Burns'],
                        mode='lines',
                        name='Power Burns',
                        line=dict(color='#ff6b35', width=2),
                        hovertemplate="Date: %{x}<br>Power Burns: %{y:.1f} Bcf/d<extra></extra>"
                    ),
                    row=1, col=1
                )
                
                # Absolute changes bar chart
                colors = ['red' if x < 0 else 'green' for x in changes_df['day_change_absolute']]
                fig.add_trace(
                    go.Bar(
                        x=pd.to_datetime(changes_df['report_date']),
                        y=changes_df['day_change_absolute'],
                        name='Daily Change',
                        marker_color=colors,
                        hovertemplate="Date: %{x}<br>Change: %{y:+.1f} Bcf/d<extra></extra>"
                    ),
                    row=2, col=1
                )
                
                # Add rolling average line for absolute changes
                if 'rolling_avg_change' in changes_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(changes_df['report_date']),
                            y=changes_df['rolling_avg_change'],
                            mode='lines',
                            name='7-Day Avg Change',
                            line=dict(color='blue', width=2, dash='dash'),
                            hovertemplate="7-Day Avg: %{y:+.1f} Bcf/d<extra></extra>"
                        ),
                        row=2, col=1
                    )
                
                # Percentage changes
                pct_colors = ['red' if x < 0 else 'green' for x in changes_df['day_change_percentage']]
                fig.add_trace(
                    go.Bar(
                        x=pd.to_datetime(changes_df['report_date']),
                        y=changes_df['day_change_percentage'],
                        name='% Change',
                        marker_color=pct_colors,
                        hovertemplate="Date: %{x}<br>Change: %{y:+.1f}%<extra></extra>"
                    ),
                    row=3, col=1
                )
                
                # Add horizontal reference lines
                fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
                fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=3, col=1)
                
                fig.update_layout(height=800, showlegend=False)
                fig.update_xaxes(title_text="Date", row=3, col=1)
                fig.update_yaxes(title_text="Power Burns (Bcf/d)", row=1, col=1)
                fig.update_yaxes(title_text="Change (Bcf/d)", row=2, col=1)
                fig.update_yaxes(title_text="Change (%)", row=3, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical summary table
                st.subheader("📈 Change Statistics Summary")
                
                stats_data = {
                    'Metric': [
                        'Average Daily Change',
                        'Median Daily Change', 
                        'Standard Deviation',
                        'Largest Single Increase',
                        'Largest Single Decrease',
                        'Days with Increases',
                        'Days with Decreases',
                        'Average % Change',
                        'Max % Increase',
                        'Max % Decrease'
                    ],
                    'Value': [
                        f"{changes_df['day_change_absolute'].mean():+.2f} Bcf/d",
                        f"{changes_df['day_change_absolute'].median():+.2f} Bcf/d",
                        f"{changes_df['day_change_absolute'].std():.2f} Bcf/d",
                        f"{changes_df['day_change_absolute'].max():+.2f} Bcf/d",
                        f"{changes_df['day_change_absolute'].min():+.2f} Bcf/d",
                        f"{(changes_df['day_change_absolute'] > 0).sum()} days",
                        f"{(changes_df['day_change_absolute'] < 0).sum()} days",
                        f"{changes_df['day_change_percentage'].mean():+.2f}%",
                        f"{changes_df['day_change_percentage'].max():+.2f}%",
                        f"{changes_df['day_change_percentage'].min():+.2f}%"
                    ]
                }
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
                
                # Recent changes highlight
                st.subheader("🔥 Recent Daily Changes (Last 10 Days)")
                recent_changes = changes_df.tail(10)[['report_date', 'L48_Power_Burns', 'day_change_absolute', 'day_change_percentage']].copy()
                recent_changes.columns = ['Date', 'Power Burns (Bcf/d)', 'Daily Change (Bcf/d)', 'Daily Change (%)']
                
                # Format the display
                recent_changes['Power Burns (Bcf/d)'] = recent_changes['Power Burns (Bcf/d)'].apply(lambda x: f"{x:.1f}")
                recent_changes['Daily Change (Bcf/d)'] = recent_changes['Daily Change (Bcf/d)'].apply(lambda x: f"{x:+.1f}")
                recent_changes['Daily Change (%)'] = recent_changes['Daily Change (%)'].apply(lambda x: f"{x:+.1f}%")
                
                st.dataframe(recent_changes, use_container_width=True)
    
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
    
    elif analysis_type == "Forecast Evolution":
        st.subheader(f"🔮 Forecast Evolution Analysis ({forecast_start_date} to {forecast_end_date})")
        
        # Load forecast evolution data
        forecast_df = get_forecast_evolution_data(forecast_start_date, forecast_end_date)
        
        if forecast_df.empty:
            st.warning("No forecast data available for the selected date range.")
        else:
            # Show available publication dates
            available_pub_dates = sorted(forecast_df['date_published'].unique())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Target Dates", f"{(forecast_end_date - forecast_start_date).days + 1}")
            with col2:
                st.metric("Publication Dates", len(available_pub_dates))
            with col3:
                st.metric("Total Forecasts", len(forecast_df))
            
            st.info(f"**Publication Dates Available:** {min(available_pub_dates)} to {max(available_pub_dates)}")
            
            # Create forecast evolution visualization
            fig = go.Figure()
            
            # Color palette for different publication dates
            colors = ['#ff6b35', '#004e89', '#009639', '#ffa400', '#9b5de5', '#f72585', '#00b4d8', '#90e0ef']
            
            # Plot each publication date as a separate line
            for i, pub_date in enumerate(available_pub_dates):
                pub_data = forecast_df[forecast_df['date_published'] == pub_date].copy()
                
                if not pub_data.empty:
                    pub_data = pub_data.sort_values('report_date')
                    
                    # Determine line style (solid for recent, dashed for older)
                    is_recent = pub_date >= max(available_pub_dates) - timedelta(days=7)
                    line_style = 'solid' if is_recent else 'dash'
                    line_width = 3 if is_recent else 2
                    
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(pub_data['report_date']),
                        y=pub_data['L48_Power_Burns'],
                        mode='lines+markers',
                        name=f"Published {pub_date}",
                        line=dict(
                            color=colors[i % len(colors)], 
                            width=line_width, 
                            dash=line_style
                        ),
                        marker=dict(size=6),
                        hovertemplate=f"Published: {pub_date}<br>Report Date: %{{x}}<br>Forecast: %{{y:.1f}} Bcf/d<extra></extra>"
                    ))
            
            fig.update_layout(
                title="Power Burns Forecast Evolution - How Forecasts Changed Over Time",
                xaxis_title="Report Date (Forecast Target Date)",
                yaxis_title="Power Burns (Bcf/d)",
                hovermode='x unified',
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast accuracy analysis for specific dates
            st.subheader("📊 Forecast Accuracy Analysis")
            
            # Select a specific target date for detailed analysis
            available_target_dates = sorted(forecast_df['report_date'].unique())
            
            selected_target_date = st.selectbox(
                "Select a specific date to analyze forecast evolution:",
                options=available_target_dates,
                index=len(available_target_dates)//2,  # Default to middle date
                help="See how the forecast for this specific date evolved over time"
            )
            
            if selected_target_date:
                target_forecasts = forecast_df[forecast_df['report_date'] == selected_target_date].copy()
                target_forecasts = target_forecasts.sort_values('date_published')
                
                if not target_forecasts.empty:
                    # Show forecast evolution for this specific date
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Scatter(
                        x=pd.to_datetime(target_forecasts['date_published']),
                        y=target_forecasts['L48_Power_Burns'],
                        mode='lines+markers',
                        name=f'Forecast for {selected_target_date}',
                        line=dict(color='#ff6b35', width=3),
                        marker=dict(size=8),
                        hovertemplate="Published: %{x}<br>Forecast: %{y:.1f} Bcf/d<extra></extra>"
                    ))
                    
                    # Add final forecast line
                    final_forecast = target_forecasts['L48_Power_Burns'].iloc[-1]
                    fig2.add_hline(
                        y=final_forecast,
                        line_dash="dot",
                        line_color="red",
                        annotation_text=f"Final Forecast: {final_forecast:.1f} Bcf/d"
                    )
                    
                    fig2.update_layout(
                        title=f"Forecast Evolution for {selected_target_date}",
                        xaxis_title="Publication Date",
                        yaxis_title="Forecasted Power Burns (Bcf/d)",
                        height=400
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Calculate forecast changes
                    target_forecasts['forecast_change'] = target_forecasts['L48_Power_Burns'].diff()
                    target_forecasts['forecast_change_pct'] = target_forecasts['L48_Power_Burns'].pct_change() * 100
                    
                    # Display forecast evolution table
                    st.subheader(f"📋 Forecast Changes for {selected_target_date}")
                    
                    display_forecasts = target_forecasts[['date_published', 'L48_Power_Burns', 'forecast_change', 'forecast_change_pct']].copy()
                    display_forecasts.columns = ['Publication Date', 'Forecast (Bcf/d)', 'Change (Bcf/d)', 'Change (%)']
                    
                    # Format display
                    display_forecasts['Forecast (Bcf/d)'] = display_forecasts['Forecast (Bcf/d)'].apply(lambda x: f"{x:.1f}")
                    display_forecasts['Change (Bcf/d)'] = display_forecasts['Change (Bcf/d)'].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "—")
                    display_forecasts['Change (%)'] = display_forecasts['Change (%)'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
                    
                    st.dataframe(display_forecasts, use_container_width=True)
                    
                    # Forecast volatility metrics
                    if len(target_forecasts) > 1:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        forecast_std = target_forecasts['L48_Power_Burns'].std()
                        forecast_range = target_forecasts['L48_Power_Burns'].max() - target_forecasts['L48_Power_Burns'].min()
                        avg_change = target_forecasts['forecast_change'].abs().mean()
                        num_revisions = len(target_forecasts) - 1
                        
                        with col1:
                            st.metric("Forecast Std Dev", f"{forecast_std:.2f} Bcf/d")
                        with col2:
                            st.metric("Forecast Range", f"{forecast_range:.2f} Bcf/d")
                        with col3:
                            st.metric("Avg Revision", f"{avg_change:.2f} Bcf/d")
                        with col4:
                            st.metric("Total Revisions", num_revisions)
            
            # Summary statistics across all dates
            st.subheader("📈 Overall Forecast Statistics")
            
            # Calculate forecast volatility for each target date
            volatility_stats = []
            
            for target_date in available_target_dates:
                date_forecasts = forecast_df[forecast_df['report_date'] == target_date]
                if len(date_forecasts) > 1:
                    volatility = date_forecasts['L48_Power_Burns'].std()
                    range_val = date_forecasts['L48_Power_Burns'].max() - date_forecasts['L48_Power_Burns'].min()
                    num_forecasts = len(date_forecasts)
                    
                    volatility_stats.append({
                        'Target Date': target_date,
                        'Num Forecasts': num_forecasts,
                        'Volatility (Std)': f"{volatility:.2f}",
                        'Range': f"{range_val:.2f}",
                        'Final Forecast': f"{date_forecasts['L48_Power_Burns'].iloc[-1]:.1f}"
                    })
            
            if volatility_stats:
                volatility_df = pd.DataFrame(volatility_stats)
                st.dataframe(volatility_df, use_container_width=True)

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