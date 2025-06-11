import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

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

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_tape_tables():
    """Get all TAPE_NG tables"""
    engine = get_db_engine()
    inspector = inspect(engine)
    all_table_names = inspector.get_table_names()
    
    tape_tables = [
        t for t in all_table_names 
        if t.startswith('TAPE_NG') and t.upper() != 'TAPE_NG'
    ]
    return sorted(tape_tables)

@st.cache_data(ttl=3600)  # Cache for 1 hour for historical data
def get_historical_tape_data(table_name: str, days_back=7):
    """Fetch last 7 days of tape data"""
    engine = get_db_engine()
    df = pd.DataFrame()
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        query = f"""
            SELECT ts_event, price, size, action, side, raw_symbol
            FROM `{table_name}` 
            WHERE ts_event >= '{start_date.strftime('%Y-%m-%d %H:%M:%S')}'
            AND ts_event <= '{end_date.strftime('%Y-%m-%d %H:%M:%S')}'
            AND action = 'T'
            ORDER BY ts_event ASC
        """
        
        df = pd.read_sql_query(query, engine)
        
        if not df.empty:
            # Convert timestamps
            df['ts_event'] = pd.to_datetime(df['ts_event'])
            
            # Convert numeric columns
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['size'] = pd.to_numeric(df['size'], errors='coerce')
            
            # Add hour grouping
            df['hour'] = df['ts_event'].dt.floor('H')
            df['date'] = df['ts_event'].dt.date
            df['hour_of_day'] = df['ts_event'].dt.hour
            
        return df
        
    except Exception as e:
        st.error(f"Error fetching historical data from table '{table_name}': {e}")
        return pd.DataFrame()

def analyze_hourly_bid_ask_summary(df):
    """
    Create simple hourly summary of whether bids or asks dominated
    """
    if df.empty or 'side' not in df.columns:
        return pd.DataFrame()
    
    # Group by hour and analyze
    hourly_summary = []
    
    for hour, group in df.groupby('hour'):
        if len(group) == 0:
            continue
        
        # Calculate volumes
        buy_volume = group[group['side'] == 'B']['size'].sum()  # Bids getting hit
        sell_volume = group[group['side'] == 'S']['size'].sum()  # Asks getting lifted
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            continue
        
        # Determine dominant side with simple logic
        bid_pct = buy_volume / total_volume
        ask_pct = sell_volume / total_volume
        
        # Simple classification
        if bid_pct > 0.6:  # 60% threshold
            dominant_side = "Bids Hit"
            dominance_strength = bid_pct
            color = "🔴"
        elif ask_pct > 0.6:
            dominant_side = "Asks Lifted"  
            dominance_strength = ask_pct
            color = "🟢"
        else:
            dominant_side = "Balanced"
            dominance_strength = 0.5
            color = "⚪"
        
        hourly_summary.append({
            'hour': hour,
            'date': hour.date(),
            'hour_of_day': hour.hour,
            'total_volume': total_volume,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'bid_pct': bid_pct,
            'ask_pct': ask_pct,
            'dominant_side': dominant_side,
            'dominance_strength': dominance_strength,
            'color': color,
            'trade_count': len(group)
        })
    
    return pd.DataFrame(hourly_summary)

def create_historical_heatmap(summary_df):
    """Create a heatmap showing bid/ask dominance by day and hour"""
    if summary_df.empty:
        return None
    
    # Create pivot table for heatmap
    # Convert dominance to numeric: Bids Hit = -1, Balanced = 0, Asks Lifted = 1
    summary_df['dominance_numeric'] = summary_df['dominant_side'].map({
        'Bids Hit': -1,
        'Balanced': 0, 
        'Asks Lifted': 1
    })
    
    # Weight by strength
    summary_df['weighted_dominance'] = (summary_df['dominance_numeric'] * 
                                       (summary_df['dominance_strength'] - 0.5) * 2)
    
    pivot_data = summary_df.pivot_table(
        values='weighted_dominance',
        index='date',
        columns='hour_of_day',
        fill_value=0
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=[f"{h:02d}:00" for h in pivot_data.columns],
        y=[str(d) for d in pivot_data.index],
        colorscale=[
            [0, 'red'],      # Strong bids hit
            [0.5, 'white'],  # Balanced  
            [1, 'green']     # Strong asks lifted
        ],
        zmid=0,
        colorbar=dict(
            title="Market Pressure",
            tickvals=[-1, 0, 1],
            ticktext=["Bids Hit", "Balanced", "Asks Lifted"]
        ),
        hovertemplate="Date: %{y}<br>Hour: %{x}<br>Pressure: %{z:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="7-Day Hourly Bid-Ask Dominance Heatmap",
        xaxis_title="Hour of Day",
        yaxis_title="Date",
        height=400
    )
    
    return fig

def create_daily_summary_chart(summary_df):
    """Create daily summary showing overall market pressure"""
    if summary_df.empty:
        return None
    
    # Daily aggregation
    daily_summary = summary_df.groupby('date').agg({
        'bid_pct': 'mean',
        'ask_pct': 'mean', 
        'total_volume': 'sum',
        'trade_count': 'sum'
    }).reset_index()
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Daily Bid vs Ask Percentage', 'Daily Volume'],
        vertical_spacing=0.1
    )
    
    # Bid/Ask percentages
    fig.add_trace(
        go.Bar(x=daily_summary['date'], y=daily_summary['bid_pct']*100,
               name='Bids Hit %', marker_color='red', opacity=0.7),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=daily_summary['date'], y=daily_summary['ask_pct']*100,
               name='Asks Lifted %', marker_color='green', opacity=0.7),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(x=daily_summary['date'], y=daily_summary['total_volume'],
               name='Total Volume', marker_color='blue'),
        row=2, col=1
    )
    
    fig.update_layout(height=500, showlegend=True)
    fig.update_yaxes(title_text="Percentage", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Historical Bid-Ask Analysis", page_icon="📅", layout="wide")

# Custom CSS
st.markdown("""
<style>
.summary-card {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    text-align: center;
}
.bids-hit {
    background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
    color: white;
}
.asks-lifted {
    background: linear-gradient(135deg, #51cf66 0%, #69db7c 100%);
    color: white;
}
.balanced {
    background: linear-gradient(135deg, #74c0fc 0%, #91c3fd 100%);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --- PAGE CONTENT ---
st.title("📅 Historical Bid-Ask Analysis")
st.markdown("---")

st.markdown("""
**7-day historical view of hourly bid-ask dynamics in Natural Gas futures.**

Simple hourly summary showing whether bids were getting hit (selling pressure) or asks were getting lifted (buying pressure).
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
        
    if st.button("📅 Bid-Ask History", use_container_width=True):
        st.switch_page("pages/6_Tape_Analysis.py")
    
    st.markdown("---")
    st.info("📍 **Current Page:** Historical Bid-Ask")
    
    st.subheader("📊 Analysis Controls")
    
    # Get available contracts
    tape_tables = get_tape_tables()
    
    if not tape_tables:
        st.error("No TAPE_NG tables found")
        st.stop()
    
    # Contract selection
    selected_table = st.selectbox(
        "Select Contract:",
        options=tape_tables,
        format_func=lambda x: x.replace('TAPE_', '').replace('_', ' '),
        help="Choose a NG futures contract for analysis"
    )
    
    # Days back selection
    days_back = st.selectbox(
        "Historical Period:",
        [3, 7, 14, 30],
        index=1,
        format_func=lambda x: f"Last {x} days",
        help="Select number of days to analyze"
    )
    
    # Real-time refresh
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Establish DB connection
engine = get_db_engine()
if not engine: 
    st.stop()

# Main analysis
if selected_table:
    
    # Load historical data
    with st.spinner(f"Loading {days_back} days of historical data for {selected_table}..."):
        df = get_historical_tape_data(selected_table, days_back)
    
    if df.empty:
        st.warning(f"No historical data found for {selected_table}")
    else:
        # Contract info
        contract_symbol = df['raw_symbol'].iloc[0] if 'raw_symbol' in df.columns else selected_table
        data_start = df['ts_event'].min()
        data_end = df['ts_event'].max()
        
        st.subheader(f"📅 {contract_symbol} - Historical Bid-Ask Summary")
        st.info(f"📊 **Data Period:** {data_start.strftime('%Y-%m-%d %H:%M')} to {data_end.strftime('%Y-%m-%d %H:%M')}")
        
        # Perform historical analysis
        with st.spinner("Analyzing historical bid-ask patterns..."):
            summary_df = analyze_hourly_bid_ask_summary(df)
        
        if summary_df.empty:
            st.error("Unable to perform bid-ask analysis. Check if 'side' data is available.")
        else:
            # Overall statistics
            total_hours = len(summary_df)
            bids_hit_hours = len(summary_df[summary_df['dominant_side'] == 'Bids Hit'])
            asks_lifted_hours = len(summary_df[summary_df['dominant_side'] == 'Asks Lifted'])
            balanced_hours = len(summary_df[summary_df['dominant_side'] == 'Balanced'])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Hours", total_hours)
            with col2:
                st.metric("🔴 Bids Hit", f"{bids_hit_hours} ({bids_hit_hours/total_hours*100:.1f}%)")
            with col3:
                st.metric("🟢 Asks Lifted", f"{asks_lifted_hours} ({asks_lifted_hours/total_hours*100:.1f}%)")
            with col4:
                st.metric("⚪ Balanced", f"{balanced_hours} ({balanced_hours/total_hours*100:.1f}%)")
            
            # Overall market sentiment
            if bids_hit_hours > asks_lifted_hours:
                sentiment = "Bearish (More Selling Pressure)"
                sentiment_color = "bids-hit"
            elif asks_lifted_hours > bids_hit_hours:
                sentiment = "Bullish (More Buying Pressure)"
                sentiment_color = "asks-lifted"
            else:
                sentiment = "Neutral (Balanced)"
                sentiment_color = "balanced"
            
            st.markdown(f"""
            <div class="summary-card {sentiment_color}">
                <h3>Overall Market Sentiment: {sentiment}</h3>
                <p>Based on {days_back}-day hourly analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Heatmap
                heatmap_fig = create_historical_heatmap(summary_df)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
            
            with col2:
                # Daily summary
                daily_fig = create_daily_summary_chart(summary_df)
                if daily_fig:
                    st.plotly_chart(daily_fig, use_container_width=True)
            
            # Hourly breakdown table
            st.subheader("📋 Hourly Breakdown")
            
            # Format for display
            display_df = summary_df.copy()
            display_df['hour_display'] = display_df['hour'].dt.strftime('%Y-%m-%d %H:00')
            display_df['bid_pct_display'] = display_df['bid_pct'].apply(lambda x: f"{x:.1%}")
            display_df['ask_pct_display'] = display_df['ask_pct'].apply(lambda x: f"{x:.1%}")
            display_df['volume_display'] = display_df['total_volume'].apply(lambda x: f"{x:,.0f}")
            
            # Create display columns
            display_columns = [
                'hour_display', 'color', 'dominant_side', 'bid_pct_display', 
                'ask_pct_display', 'volume_display', 'trade_count'
            ]
            
            # Show recent data first
            display_df = display_df.sort_values('hour', ascending=False)
            
            st.dataframe(
                display_df[display_columns].head(50),  # Show last 50 hours
                use_container_width=True,
                column_config={
                    'hour_display': 'Hour',
                    'color': 'Status',
                    'dominant_side': 'Market Action',
                    'bid_pct_display': 'Bids Hit %',
                    'ask_pct_display': 'Asks Lifted %',
                    'volume_display': 'Total Volume',
                    'trade_count': st.column_config.NumberColumn('Trades', format="%.0f")
                }
            )
            
            # Export functionality
            if st.button("📥 Download Historical Analysis"):
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{contract_symbol}_historical_bid_ask_{days_back}days.csv",
                    mime="text/csv"
                )

st.markdown("---")
st.markdown(f"**💾 Data Source:** Historical Tape Data | **📅 Period:** Last {days_back if 'days_back' in locals() else 7} days | **⏰ Resolution:** Hourly")