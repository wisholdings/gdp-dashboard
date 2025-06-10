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
            
            # Add year and day of year for seasonal analysis
            df['year'] = pd.to_datetime(df['report_date']).dt.year
            df['day_of_year'] = pd.to_datetime(df['report_date']).dt.dayofyear
            
        return df
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_data_date_range():
    """Get the available date range from the database"""
    engine = get_db_engine()
    try:
        query = "SELECT MIN(report_date) as min_date, MAX(report_date) as max_date FROM power_burns_daily"
        result = pd.read_sql_query(query, engine)
        if not result.empty:
            min_date = pd.to_datetime(result['min_date'].iloc[0]).date()
            max_date = pd.to_datetime(result['max_date'].iloc[0]).date()
            return min_date, max_date
        return None, None
    except Exception as e:
        st.error(f"Error fetching date range: {e}")
        return None, None

@st.cache_data(ttl=3600)
def get_available_publication_dates():
    """Get available publication dates from the database"""
    engine = get_db_engine()
    try:
        query = "SELECT DISTINCT date_published FROM power_burns_daily ORDER BY date_published DESC LIMIT 10"
        df = pd.read_sql_query(query, engine)
        if not df.empty:
            return [pd.to_datetime(date).date() for date in df['date_published']]
        return []
    except Exception as e:
        st.error(f"Error fetching publication dates: {e}")
        return []

def calculate_seasonal_stats(df):
    """Calculate seasonal statistics"""
    if df.empty:
        return pd.DataFrame()
    
    # Add season column
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['month'] = pd.to_datetime(df['report_date']).dt.month
    df['season'] = df['month'].apply(get_season)
    
    seasonal_stats = df.groupby('season').agg({
        'L48_Power_Burns': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    
    seasonal_stats.columns = ['Avg_Burns', 'Std_Dev', 'Min_Burns', 'Max_Burns', 'Data_Points']
    
    return seasonal_stats.reset_index()

def calculate_yearly_stats(df):
    """Calculate yearly statistics"""
    if df.empty:
        return pd.DataFrame()
    
    yearly_stats = df.groupby('year').agg({
        'L48_Power_Burns': ['mean', 'sum', 'std', 'min', 'max', 'count']
    }).round(2)
    
    yearly_stats.columns = ['Avg_Daily_Burns', 'Total_Annual_Burns', 'Std_Dev', 'Min_Burns', 'Max_Burns', 'Data_Points']
    
    return yearly_stats.reset_index()

def calculate_forecast_run_changes(df, selected_pub_dates):
    """Calculate changes between different forecast runs - ENHANCED WITH CUMULATIVE"""
    if df.empty or len(selected_pub_dates) < 2:
        return pd.DataFrame()
    
    # Convert date_published to date for comparison
    df['pub_date'] = df['date_published'].dt.date
    
    # Get data for each selected publication date
    runs_data = {}
    for pub_date in selected_pub_dates:
        run_data = df[df['pub_date'] == pub_date].copy()
        if not run_data.empty:
            runs_data[pub_date] = run_data.set_index('report_date')['L48_Power_Burns']
    
    if len(runs_data) < 2:
        return pd.DataFrame()
    
    # Calculate changes between runs (newest vs oldest, or consecutive if more than 2)
    sorted_dates = sorted(selected_pub_dates)
    
    if len(sorted_dates) == 2:
        old_run = runs_data[sorted_dates[0]]
        new_run = runs_data[sorted_dates[1]]
        
        # Align data by report_date
        combined = pd.DataFrame({
            'old_forecast': old_run,
            'new_forecast': new_run
        }).dropna()
        
        combined['absolute_change'] = combined['new_forecast'] - combined['old_forecast']
        combined['percentage_change'] = (combined['absolute_change'] / combined['old_forecast']) * 100
        
        # *** NEW: Calculate cumulative change ***
        combined['cumulative_change'] = combined['absolute_change'].cumsum()
        
        return combined.reset_index()
    
    # For multiple runs, calculate consecutive changes
    changes_list = []
    for i in range(1, len(sorted_dates)):
        old_run = runs_data[sorted_dates[i-1]]
        new_run = runs_data[sorted_dates[i]]
        
        combined = pd.DataFrame({
            'old_forecast': old_run,
            'new_forecast': new_run
        }).dropna()
        
        combined['absolute_change'] = combined['new_forecast'] - combined['old_forecast']
        combined['percentage_change'] = (combined['absolute_change'] / combined['old_forecast']) * 100
        combined['comparison'] = f"{sorted_dates[i-1]} vs {sorted_dates[i]}"
        
        # *** NEW: Calculate cumulative change ***
        combined['cumulative_change'] = combined['absolute_change'].cumsum()
        
        changes_list.append(combined.reset_index())
    
    return pd.concat(changes_list, ignore_index=True)

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
                "Start Date:",
                value=default_start,
                min_value=min_date,
                max_value=max_date,
                help="Starting date for analysis"
            )
            
            end_date = st.date_input(
                "End Date:",
                value=default_end,
                min_value=min_date,
                max_value=max_date,
                help="Ending date for analysis"
            )
            
            if start_date > end_date:
                st.error("Start date must be before end date")
                st.stop()
                
        elif analysis_type == "Day-over-Day Changes":
            st.markdown("**📊 Compare Forecast Runs**")
            
            # Get available publication dates
            pub_dates = get_available_publication_dates()
            if not pub_dates:
                st.error("No publication dates found in database")
                st.stop()
            
            # Let user select publication dates to compare
            selected_pub_dates = st.multiselect(
                "Select Publication Dates to Compare:",
                options=pub_dates,
                default=pub_dates[:2] if len(pub_dates) >= 2 else pub_dates,
                help="Choose 2 or more publication dates to compare their forecasts"
            )
            
            # Date range for the comparison
            dod_start_date = st.date_input(
                "Report Period Start:",
                value=max(min_date, datetime.now().date() - timedelta(days=7)),
                min_value=min_date,
                max_value=max_date,
                help="Starting date for the forecast period to compare"
            )
            
            dod_end_date = st.date_input(
                "Report Period End:",
                value=min(max_date, datetime.now().date() + timedelta(days=14)),
                min_value=min_date,
                max_value=max_date,
                help="Ending date for the forecast period to compare"
            )
            
            if dod_start_date > dod_end_date:
                st.error("Start date must be before end date")
                st.stop()
                
        elif analysis_type == "Seasonal Analysis":
            # Get available years
            df_temp = get_power_burns_data()
            if not df_temp.empty:
                available_years = sorted(df_temp['year'].unique())
                selected_years = st.multiselect(
                    "Select Years for Comparison:",
                    options=available_years,
                    default=available_years[-3:] if len(available_years) >= 3 else available_years,
                    help="Choose years to compare seasonal patterns"
                )
            else:
                st.error("No data available for seasonal analysis")
                st.stop()
                
        elif analysis_type == "Yearly Comparison":
            # Similar to seasonal but focus on yearly stats
            df_temp = get_power_burns_data()
            if not df_temp.empty:
                available_years = sorted(df_temp['year'].unique())
                selected_years = st.multiselect(
                    "Select Years for Statistics:",
                    options=available_years,
                    default=available_years,
                    help="Choose years to include in yearly statistics"
                )
            else:
                st.error("No data available for yearly comparison")
                st.stop()
                
        elif analysis_type == "Forecast Evolution":
            st.markdown("**🔮 Track Forecast Changes**")
            
            # Target dates you want to see forecasts for
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
        st.subheader(f"📊 Forecast Run Comparison ({dod_start_date} to {dod_end_date})")
        
        # Load data for selected date range
        df = get_power_burns_data(dod_start_date, dod_end_date)
        
        if df.empty:
            st.warning("No data available for forecast comparison in the selected date range.")
        elif not selected_pub_dates:
            st.warning("Please select at least one publication date.")
        elif len(selected_pub_dates) < 2:
            st.warning("Please select at least 2 publication dates to compare forecasts.")
        else:
            # Filter data for selected publication dates
            df['pub_date'] = df['date_published'].dt.date
            filtered_df = df[df['pub_date'].isin(selected_pub_dates)]
            
            if filtered_df.empty:
                st.warning("No data found for the selected publication dates.")
            else:
                # Calculate forecast changes
                changes_df = calculate_forecast_run_changes(filtered_df, selected_pub_dates)
                
                if changes_df.empty:
                    st.warning("Unable to calculate forecast changes. Make sure the selected runs have overlapping report dates.")
                else:
                    # Sort publication dates for clear labeling
                    sorted_pub_dates = sorted(selected_pub_dates)
                    old_date = sorted_pub_dates[0]
                    new_date = sorted_pub_dates[-1]
                    
                    # *** ENHANCED METRICS DISPLAY WITH CUMULATIVE ***
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    avg_change = changes_df['absolute_change'].mean()
                    max_increase = changes_df['absolute_change'].max()
                    max_decrease = changes_df['absolute_change'].min()
                    volatility = changes_df['absolute_change'].std()
                    # *** NEW: Cumulative effect for the time period ***
                    cumulative_effect = changes_df['absolute_change'].sum()
                    
                    with col1:
                        st.metric("Avg Forecast Change", f"{avg_change:+.1f} Bcf/d")
                    with col2:
                        st.metric("Largest Increase", f"{max_increase:+.1f} Bcf/d")
                    with col3:
                        st.metric("Largest Decrease", f"{max_decrease:+.1f} Bcf/d")
                    with col4:
                        st.metric("Change Volatility", f"{volatility:.1f} Bcf/d")
                    with col5:
                        # *** NEW METRIC ***
                        st.metric("Cumulative Effect", f"{cumulative_effect:+.1f} Bcf/d", 
                                 help="Total sum of all forecast changes across the time period")
                    
                    # Clear labeling of what's being compared
                    st.info(f"**📅 Comparing:** {old_date} (Old Forecast) vs {new_date} (New Forecast)")
                    
                    # *** ENHANCED VISUALIZATION WITH CUMULATIVE PLOT ***
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=[
                            f'Forecast Comparison: {old_date} vs {new_date}',
                            'Daily Forecast Changes (New - Old)',
                            'Cumulative Effect Over Time'
                        ],
                        vertical_spacing=0.12,
                        row_heights=[0.4, 0.3, 0.3]
                    )
                    
                    # Top plot: Line charts comparing forecasts
                    fig.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(changes_df['report_date']),
                            y=changes_df['old_forecast'],
                            mode='lines+markers',
                            name=f'Old ({old_date})',
                            line=dict(color='#004e89', width=3),
                            marker=dict(size=6),
                            hovertemplate=f"Old Forecast ({old_date})<br>Date: %{{x}}<br>Power Burns: %{{y:.1f}} Bcf/d<extra></extra>"
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(changes_df['report_date']),
                            y=changes_df['new_forecast'],
                            mode='lines+markers',
                            name=f'New ({new_date})',
                            line=dict(color='#ff6b35', width=3),
                            marker=dict(size=6),
                            hovertemplate=f"New Forecast ({new_date})<br>Date: %{{x}}<br>Power Burns: %{{y:.1f}} Bcf/d<extra></extra>"
                        ),
                        row=1, col=1
                    )
                    
                    # Middle plot: Bar chart of daily changes
                    colors = ['red' if x < 0 else 'green' for x in changes_df['absolute_change']]
                    fig.add_trace(
                        go.Bar(
                            x=pd.to_datetime(changes_df['report_date']),
                            y=changes_df['absolute_change'],
                            name='Daily Change',
                            marker_color=colors,
                            hovertemplate="Date: %{x}<br>Daily Change: %{y:+.1f} Bcf/d<extra></extra>",
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    
                    # *** NEW: Bottom plot: Cumulative effect line chart ***
                    fig.add_trace(
                        go.Scatter(
                            x=pd.to_datetime(changes_df['report_date']),
                            y=changes_df['cumulative_change'],
                            mode='lines+markers',
                            name='Cumulative Effect',
                            line=dict(color='purple', width=3),
                            marker=dict(size=6),
                            fill='tonexty' if cumulative_effect > 0 else 'tozeroy',
                            fillcolor='rgba(128, 0, 128, 0.2)',
                            hovertemplate="Date: %{x}<br>Cumulative Effect: %{y:+.1f} Bcf/d<extra></extra>",
                            showlegend=False
                        ),
                        row=3, col=1
                    )
                    
                    # Add horizontal reference lines
                    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=2, col=1)
                    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=3, col=1)
                    
                    fig.update_layout(height=900, showlegend=True)
                    fig.update_xaxes(title_text="Date", row=3, col=1)
                    fig.update_yaxes(title_text="Power Burns (Bcf/d)", row=1, col=1)
                    fig.update_yaxes(title_text="Daily Change (Bcf/d)", row=2, col=1)
                    fig.update_yaxes(title_text="Cumulative Change (Bcf/d)", row=3, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # *** ENHANCED SUMMARY TABLE ***
                    st.subheader("📊 Enhanced Forecast Change Summary")
                    
                    display_changes = changes_df[['report_date', 'old_forecast', 'new_forecast', 
                                                 'absolute_change', 'percentage_change', 'cumulative_change']].copy()
                    display_changes.columns = ['Date', f'Old Forecast ({old_date})', f'New Forecast ({new_date})', 
                                              'Daily Change (Bcf/d)', 'Daily Change (%)', 'Cumulative Effect (Bcf/d)']
                    
                    # Format display
                    display_changes[f'Old Forecast ({old_date})'] = display_changes[f'Old Forecast ({old_date})'].apply(lambda x: f"{x:.1f}")
                    display_changes[f'New Forecast ({new_date})'] = display_changes[f'New Forecast ({new_date})'].apply(lambda x: f"{x:.1f}")
                    display_changes['Daily Change (Bcf/d)'] = display_changes['Daily Change (Bcf/d)'].apply(lambda x: f"{x:+.1f}")
                    display_changes['Daily Change (%)'] = display_changes['Daily Change (%)'].apply(lambda x: f"{x:+.1f}%")
                    # *** NEW COLUMN ***
                    display_changes['Cumulative Effect (Bcf/d)'] = display_changes['Cumulative Effect (Bcf/d)'].apply(lambda x: f"{x:+.1f}")
                    
                    st.dataframe(display_changes, use_container_width=True)
                    
                    # *** NEW: Summary insights box ***
                    st.info(f"""
                    **📈 Key Insights for Time Period:**
                    - **Average Daily Change:** {avg_change:+.1f} Bcf/d
                    - **Total Cumulative Effect:** {cumulative_effect:+.1f} Bcf/d over {len(changes_df)} days
                    - **Net Direction:** {'Upward revision' if cumulative_effect > 0 else 'Downward revision' if cumulative_effect < 0 else 'Neutral revision'}
                    - **Period Impact:** {'Increasing burns forecast' if cumulative_effect > 0 else 'Decreasing burns forecast' if cumulative_effect < 0 else 'No net change in forecast'}
                    """)

    elif analysis_type == "Seasonal Analysis":
        st.subheader("🌿 Seasonal Power Burns Analysis")
        
        # Load all data for seasonal analysis
        df = get_power_burns_data()
        
        if df.empty:
            st.warning("No data available for seasonal analysis.")
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
            
            # Seasonal statistics
            seasonal_stats = calculate_seasonal_stats(df_filtered)
            st.subheader("📊 Seasonal Statistics")
            st.dataframe(seasonal_stats, use_container_width=True)
            
            # Yearly statistics
            yearly_stats = calculate_yearly_stats(df_filtered)
            st.subheader("📊 Yearly Statistics")
            st.dataframe(yearly_stats, use_container_width=True)
    
    elif analysis_type == "Yearly Comparison":
        st.subheader("📅 Yearly Power Burns Statistics")
        
        # Load all data
        df = get_power_burns_data()
        
        if df.empty:
            st.warning("No data available for yearly comparison.")
        else:
            # Filter for selected years
            df_filtered = df[df['year'].isin(selected_years)]
            
            # Calculate yearly statistics
            yearly_stats = calculate_yearly_stats(df_filtered)
            
            # Display statistics table
            st.subheader("📊 Yearly Statistics Summary")
            st.dataframe(yearly_stats, use_container_width=True)
            
            # Create comparison charts
            if len(selected_years) > 1:
                # Annual average comparison
                fig_avg = go.Figure()
                fig_avg.add_trace(go.Bar(
                    x=yearly_stats['year'],
                    y=yearly_stats['Avg_Daily_Burns'],
                    name='Average Daily Burns',
                    marker_color='#ff6b35',
                    text=[f"{val:.1f}" for val in yearly_stats['Avg_Daily_Burns']],
                    textposition='auto'
                ))
                
                fig_avg.update_layout(
                    title="Average Daily Power Burns by Year",
                    xaxis_title="Year",
                    yaxis_title="Average Daily Burns (Bcf/d)",
                    height=400
                )
                
                st.plotly_chart(fig_avg, use_container_width=True)
                
                # Volatility comparison
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=yearly_stats['year'],
                    y=yearly_stats['Std_Dev'],
                    name='Standard Deviation',
                    marker_color='#004e89',
                    text=[f"{val:.1f}" for val in yearly_stats['Std_Dev']],
                    textposition='auto'
                ))
                
                fig_vol.update_layout(
                    title="Power Burns Volatility by Year",
                    xaxis_title="Year",
                    yaxis_title="Standard Deviation (Bcf/d)",
                    height=400
                )
                
                st.plotly_chart(fig_vol, use_container_width=True)
    
    elif analysis_type == "Forecast Evolution":
        st.subheader(f"🔮 Forecast Evolution Analysis ({forecast_start_date} to {forecast_end_date})")
        
        # Load forecast evolution data
        forecast_df = get_forecast_evolution_data(forecast_start_date, forecast_end_date)
        
        if forecast_df.empty:
            st.warning("No forecast data available for the selected date range.")
        else:
            # Get unique target dates and publication dates
            target_dates = sorted(forecast_df['report_date'].unique())
            publication_dates = sorted(forecast_df['date_published'].unique())
            
            st.info(f"**Found forecasts for {len(target_dates)} target dates from {len(publication_dates)} publication dates**")
            
            # Select specific target dates to analyze
            available_target_dates = target_dates[:10]  # Limit to first 10 for performance
            
            selected_target_dates = st.multiselect(
                "Select Target Dates to Analyze:",
                options=available_target_dates,
                default=available_target_dates[:3] if len(available_target_dates) >= 3 else available_target_dates,
                help="Choose specific dates to see how their forecasts evolved over time"
            )
            
            if selected_target_dates:
                # Create evolution plot
                fig = go.Figure()
                
                colors = ['#ff6b35', '#004e89', '#009639', '#ffa400', '#9b5de5', '#f72585']
                
                for i, target_date in enumerate(selected_target_dates):
                    target_data = forecast_df[forecast_df['report_date'] == target_date].copy()
                    target_data = target_data.sort_values('date_published')
                    
                    if not target_data.empty:
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(target_data['date_published']),
                            y=target_data['L48_Power_Burns'],
                            mode='lines+markers',
                            name=f'{target_date}',
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=6),
                            hovertemplate=f"Target: {target_date}<br>Published: %{{x}}<br>Forecast: %{{y:.1f}} Bcf/d<extra></extra>"
                        ))
                
                fig.update_layout(
                    title="How Forecasts Evolved Over Time",
                    xaxis_title="Publication Date",
                    yaxis_title="Forecasted Power Burns (Bcf/d)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
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