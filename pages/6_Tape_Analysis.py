import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
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

@st.cache_data(ttl=600)  # Cache for 10 minutes due to high-frequency data
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

@st.cache_data(ttl=300)  # 5-minute cache for tape data
def get_tape_data(table_name: str, limit=50000):
    """Fetch tape data with limit for performance"""
    engine = get_db_engine()
    df = pd.DataFrame()
    
    try:
        # Get recent data with limit for performance
        query = f"""
            SELECT ts_event, instrument_id, price, size, action, side, 
                   ts_definition_event, raw_symbol, expiration, 
                   min_price_increment, display_factor, contract_multiplier,
                   currency, exchange, asset, unit_of_measure, trading_reference_price
            FROM `{table_name}` 
            ORDER BY ts_event DESC 
            LIMIT {limit}
        """
        
        df = pd.read_sql_query(query, engine)
        
        if not df.empty:
            # Convert timestamps
            df['ts_event'] = pd.to_datetime(df['ts_event'])
            df['ts_definition_event'] = pd.to_datetime(df['ts_definition_event'])
            df['expiration'] = pd.to_datetime(df['expiration'])
            
            # Convert numeric columns
            numeric_cols = ['price', 'size', 'min_price_increment', 'display_factor', 
                          'contract_multiplier', 'trading_reference_price']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp ascending for analysis
            df = df.sort_values('ts_event').reset_index(drop=True)
            
            # Add time grouping columns
            df['millisecond'] = df['ts_event'].dt.floor('L')
            df['second'] = df['ts_event'].dt.floor('S')
            df['minute'] = df['ts_event'].dt.floor('T')
            df['hour'] = df['ts_event'].dt.floor('H')
            df['time_5min'] = df['ts_event'].dt.floor('5T')
            df['time_15min'] = df['ts_event'].dt.floor('15T')
            df['time_30min'] = df['ts_event'].dt.floor('30T')
            
            # Add notional value
            df['notional'] = df['price'] * df['size'] * df['contract_multiplier']
            
        return df
        
    except Exception as e:
        st.error(f"Error fetching data from table '{table_name}': {e}")
        return pd.DataFrame()

def analyze_bid_ask_dynamics(df, time_grouping='hour'):
    """
    Analyze bid-ask dynamics to determine if bids are getting hit or asks are getting lifted
    
    Logic:
    - When side='B' (Buy), this typically means someone is hitting the bid (selling at bid price)
    - When side='S' (Sell), this typically means someone is lifting the ask (buying at ask price)
    - We'll analyze volume patterns and price movements to confirm this
    """
    if df.empty:
        return pd.DataFrame()
    
    # Filter for trades only
    trades = df[df['action'] == 'T'].copy() if 'action' in df.columns else df.copy()
    
    if trades.empty or 'side' not in trades.columns:
        return pd.DataFrame()
    
    # Group by the specified time period
    time_col = time_grouping
    if time_col not in trades.columns:
        time_col = 'hour'  # fallback
    
    # Analyze bid-ask dynamics by time period
    bid_ask_analysis = []
    
    for time_period, group in trades.groupby(time_col):
        if len(group) == 0:
            continue
            
        # Basic metrics
        total_volume = group['size'].sum()
        total_trades = len(group)
        avg_price = group['price'].mean()
        price_range = group['price'].max() - group['price'].min()
        
        # Bid hits vs Ask lifts analysis
        # In market convention:
        # - Buy side ('B') often indicates hitting the bid (aggressive sell)
        # - Sell side ('S') often indicates lifting the ask (aggressive buy)
        # But this can vary by data provider, so we'll analyze both ways
        
        buy_side_trades = group[group['side'] == 'B']
        sell_side_trades = group[group['side'] == 'S']
        
        buy_volume = buy_side_trades['size'].sum() if not buy_side_trades.empty else 0
        sell_volume = sell_side_trades['size'].sum() if not sell_side_trades.empty else 0
        
        buy_count = len(buy_side_trades)
        sell_count = len(sell_side_trades)
        
        # Calculate average prices for each side
        buy_avg_price = buy_side_trades['price'].mean() if not buy_side_trades.empty else 0
        sell_avg_price = sell_side_trades['price'].mean() if not sell_side_trades.empty else 0
        
        # Determine market pressure
        volume_imbalance = (buy_volume - sell_volume) / max(total_volume, 1)
        trade_imbalance = (buy_count - sell_count) / max(total_trades, 1)
        
        # Price movement analysis (compared to period start)
        price_change = group['price'].iloc[-1] - group['price'].iloc[0] if len(group) > 1 else 0
        price_change_pct = (price_change / group['price'].iloc[0] * 100) if group['price'].iloc[0] != 0 else 0
        
        # Determine dominant market action
        if buy_volume > sell_volume * 1.2:  # 20% threshold
            market_action = "Bids Getting Hit"
            action_confidence = abs(volume_imbalance)
        elif sell_volume > buy_volume * 1.2:
            market_action = "Asks Getting Lifted"  
            action_confidence = abs(volume_imbalance)
        else:
            market_action = "Balanced"
            action_confidence = 0.5
        
        # Calculate intensity metrics
        volume_per_minute = total_volume / max((group['ts_event'].max() - group['ts_event'].min()).total_seconds() / 60, 1)
        trades_per_minute = total_trades / max((group['ts_event'].max() - group['ts_event'].min()).total_seconds() / 60, 1)
        
        bid_ask_analysis.append({
            'time_period': time_period,
            'total_volume': total_volume,
            'total_trades': total_trades,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'buy_avg_price': buy_avg_price,
            'sell_avg_price': sell_avg_price,
            'volume_imbalance': volume_imbalance,
            'trade_imbalance': trade_imbalance,
            'market_action': market_action,
            'action_confidence': action_confidence,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'avg_price': avg_price,
            'price_range': price_range,
            'volume_per_minute': volume_per_minute,
            'trades_per_minute': trades_per_minute,
            'period_start': group['ts_event'].min(),
            'period_end': group['ts_event'].max()
        })
    
    return pd.DataFrame(bid_ask_analysis)

def create_bid_ask_visualizations(analysis_df):
    """Create comprehensive bid-ask analysis visualizations"""
    if analysis_df.empty:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            'Volume: Bids Hit vs Asks Lifted', 'Market Action Distribution',
            'Volume Imbalance Over Time', 'Price Changes vs Market Action',
            'Trading Intensity (Volume/Min)', 'Trade Count Imbalance',
            'Price Movement vs Volume Imbalance', 'Confidence Levels'
        ],
        specs=[[{"secondary_y": False}, {"type": "pie"}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.08
    )
    
    # 1. Volume comparison (Buy vs Sell)
    fig.add_trace(
        go.Bar(x=analysis_df['time_period'], y=analysis_df['buy_volume'], 
               name='Buy Volume (Bids Hit)', marker_color='red', opacity=0.7),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=analysis_df['time_period'], y=analysis_df['sell_volume'], 
               name='Sell Volume (Asks Lifted)', marker_color='green', opacity=0.7),
        row=1, col=1
    )
    
    # 2. Market action pie chart
    action_counts = analysis_df['market_action'].value_counts()
    fig.add_trace(
        go.Pie(labels=action_counts.index, values=action_counts.values,
               name="Market Action", marker_colors=['red', 'green', 'blue']),
        row=1, col=2
    )
    
    # 3. Volume imbalance over time
    colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in analysis_df['volume_imbalance']]
    fig.add_trace(
        go.Bar(x=analysis_df['time_period'], y=analysis_df['volume_imbalance'],
               name='Volume Imbalance', marker_color=colors),
        row=2, col=1
    )
    
    # 4. Price changes vs market action
    action_colors = {'Bids Getting Hit': 'red', 'Asks Getting Lifted': 'green', 'Balanced': 'blue'}
    for action in analysis_df['market_action'].unique():
        action_data = analysis_df[analysis_df['market_action'] == action]
        fig.add_trace(
            go.Scatter(x=action_data['time_period'], y=action_data['price_change_pct'],
                      mode='markers', name=f'Price Change - {action}',
                      marker=dict(color=action_colors.get(action, 'gray'), size=8)),
            row=2, col=2
        )
    
    # 5. Trading intensity
    fig.add_trace(
        go.Scatter(x=analysis_df['time_period'], y=analysis_df['volume_per_minute'],
                  mode='lines+markers', name='Volume/Min', line=dict(color='purple')),
        row=3, col=1
    )
    
    # 6. Trade count imbalance
    fig.add_trace(
        go.Bar(x=analysis_df['time_period'], y=analysis_df['trade_imbalance'],
               name='Trade Count Imbalance', marker_color='orange'),
        row=3, col=2
    )
    
    # 7. Price movement vs volume imbalance (scatter)
    fig.add_trace(
        go.Scatter(x=analysis_df['volume_imbalance'], y=analysis_df['price_change_pct'],
                  mode='markers', name='Price vs Volume Imbalance',
                  marker=dict(size=analysis_df['total_volume']/1000, color='teal', opacity=0.6)),
        row=4, col=1
    )
    
    # 8. Confidence levels
    fig.add_trace(
        go.Bar(x=analysis_df['time_period'], y=analysis_df['action_confidence'],
               name='Action Confidence', marker_color='lightblue'),
        row=4, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text="Comprehensive Bid-Ask Dynamics Analysis"
    )
    
    # Add horizontal line at zero for imbalance charts
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=2)
    
    return fig

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Bid-Ask Analysis", page_icon="⚖️", layout="wide")

# Custom CSS
st.markdown("""
<style>
.bid-hit {
    background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.ask-lifted {
    background: linear-gradient(135deg, #51cf66 0%, #69db7c 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.balanced {
    background: linear-gradient(135deg, #74c0fc 0%, #91c3fd 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# --- PAGE CONTENT ---
st.title("⚖️ Bid-Ask Dynamics Analysis")
st.markdown("---")

st.markdown("""
**Real-time analysis of bid-ask dynamics in Natural Gas futures.**

This analysis identifies whether market participants are primarily hitting bids (selling pressure) 
or lifting asks (buying pressure) in each time period.
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
        
    if st.button("⚖️ Bid-Ask Analysis", use_container_width=True):
        st.switch_page("pages/6_Tape_Analysis.py")
    
    st.markdown("---")
    st.info("📍 **Current Page:** Bid-Ask Analysis")
    
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
    
    # Time slice selection
    time_slice = st.selectbox(
        "Time Slice:",
        ["hour", "time_30min", "time_15min", "time_5min", "minute"],
        format_func=lambda x: {
            "hour": "1 Hour", 
            "time_30min": "30 Minutes",
            "time_15min": "15 Minutes", 
            "time_5min": "5 Minutes",
            "minute": "1 Minute"
        }.get(x, x),
        help="Select time period for bid-ask analysis"
    )
    
    # Data limit for performance
    data_limit = st.selectbox(
        "Data Points:",
        [10000, 25000, 50000, 100000],
        index=2,
        help="Number of recent records to analyze"
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
    
    # Load data
    with st.spinner(f"Loading tape data for {selected_table}..."):
        df = get_tape_data(selected_table, data_limit)
    
    if df.empty:
        st.warning(f"No data found for {selected_table}")
    else:
        # Contract info
        contract_symbol = df['raw_symbol'].iloc[0] if 'raw_symbol' in df.columns else selected_table
        expiration = df['expiration'].iloc[0] if 'expiration' in df.columns else None
        
        st.subheader(f"⚖️ {contract_symbol} - Bid-Ask Dynamics")
        
        # Perform bid-ask analysis
        with st.spinner("Analyzing bid-ask dynamics..."):
            analysis_df = analyze_bid_ask_dynamics(df, time_slice)
        
        if analysis_df.empty:
            st.error("Unable to perform bid-ask analysis. Check if 'side' data is available.")
        else:
            # Summary metrics
            total_periods = len(analysis_df)
            bids_hit_periods = len(analysis_df[analysis_df['market_action'] == 'Bids Getting Hit'])
            asks_lifted_periods = len(analysis_df[analysis_df['market_action'] == 'Asks Getting Lifted'])
            balanced_periods = len(analysis_df[analysis_df['market_action'] == 'Balanced'])
            
            # Overall statistics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Periods", total_periods)
            with col2:
                st.metric("Bids Hit", f"{bids_hit_periods} ({bids_hit_periods/total_periods*100:.1f}%)")
            with col3:
                st.metric("Asks Lifted", f"{asks_lifted_periods} ({asks_lifted_periods/total_periods*100:.1f}%)")
            with col4:
                st.metric("Balanced", f"{balanced_periods} ({balanced_periods/total_periods*100:.1f}%)")
            with col5:
                avg_confidence = analysis_df['action_confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            
            if expiration:
                days_to_expiry = (expiration.date() - datetime.now().date()).days
                st.info(f"⏰ **Contract Expiration:** {expiration.date()} ({days_to_expiry} days remaining)")
            
            # Market sentiment analysis
            st.subheader("📊 Market Sentiment Summary")
            
            latest_period = analysis_df.iloc[-1] if not analysis_df.empty else None
            if latest_period is not None:
                action = latest_period['market_action']
                confidence = latest_period['action_confidence']
                volume_imbalance = latest_period['volume_imbalance']
                
                if action == "Bids Getting Hit":
                    st.markdown(f"""
                    <div class="bid-hit">
                        <h3>🔴 Bids Getting Hit (Selling Pressure)</h3>
                        <p>Volume Imbalance: {volume_imbalance:.1%} | Confidence: {confidence:.1%}</p>
                        <p>Market showing selling pressure with {latest_period['buy_volume']:,.0f} volume hitting bids vs {latest_period['sell_volume']:,.0f} lifting asks</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif action == "Asks Getting Lifted":
                    st.markdown(f"""
                    <div class="ask-lifted">
                        <h3>🟢 Asks Getting Lifted (Buying Pressure)</h3>
                        <p>Volume Imbalance: {volume_imbalance:.1%} | Confidence: {confidence:.1%}</p>
                        <p>Market showing buying pressure with {latest_period['sell_volume']:,.0f} volume lifting asks vs {latest_period['buy_volume']:,.0f} hitting bids</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="balanced">
                        <h3>⚪ Balanced Market</h3>
                        <p>Volume Imbalance: {volume_imbalance:.1%} | Confidence: {confidence:.1%}</p>
                        <p>Market showing balanced activity between bid hits and ask lifts</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Create visualizations
            fig = create_bid_ask_visualizations(analysis_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("📋 Detailed Bid-Ask Analysis")
            
            # Format the dataframe for display
            display_df = analysis_df.copy()
            display_df['time_period'] = display_df['time_period'].dt.strftime('%Y-%m-%d %H:%M')
            display_df['volume_imbalance'] = display_df['volume_imbalance'].apply(lambda x: f"{x:.1%}")
            display_df['action_confidence'] = display_df['action_confidence'].apply(lambda x: f"{x:.1%}")
            display_df['price_change_pct'] = display_df['price_change_pct'].apply(lambda x: f"{x:.2f}%")
            
            # Select columns to display
            display_columns = [
                'time_period', 'market_action', 'total_volume', 'buy_volume', 'sell_volume',
                'volume_imbalance', 'action_confidence', 'price_change_pct', 'trades_per_minute'
            ]
            
            st.dataframe(
                display_df[display_columns].round(2),
                use_container_width=True,
                column_config={
                    'time_period': 'Time Period',
                    'market_action': 'Market Action',
                    'total_volume': st.column_config.NumberColumn('Total Volume', format="%.0f"),
                    'buy_volume': st.column_config.NumberColumn('Buy Volume', format="%.0f"),
                    'sell_volume': st.column_config.NumberColumn('Sell Volume', format="%.0f"),
                    'volume_imbalance': 'Volume Imbalance',
                    'action_confidence': 'Confidence',
                    'price_change_pct': 'Price Change %',
                    'trades_per_minute': st.column_config.NumberColumn('Trades/Min', format="%.1f")
                }
            )
            
            # Export functionality
            if st.button("📥 Download Analysis"):
                csv = analysis_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{contract_symbol}_bid_ask_analysis_{time_slice}.csv",
                    mime="text/csv"
                )

st.markdown("---")
st.markdown("**💾 Data Source:** High-Frequency Tape Data | **🔄 Updates:** Real-time | **📊 Analysis:** Bid-Ask Dynamics")