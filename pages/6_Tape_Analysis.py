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
            
            # Add derived columns for analysis
            df['millisecond'] = df['ts_event'].dt.floor('L')  # Round to millisecond
            df['second'] = df['ts_event'].dt.floor('S')  # Round to second
            df['minute'] = df['ts_event'].dt.floor('T')  # Round to minute
            df['hour'] = df['ts_event'].dt.floor('H')  # Round to hour
            
            # Add notional value
            df['notional'] = df['price'] * df['size'] * df['contract_multiplier']
            
        return df
        
    except Exception as e:
        st.error(f"Error fetching data from table '{table_name}': {e}")
        return pd.DataFrame()

def calculate_market_microstructure_metrics(df):
    """Calculate advanced market microstructure metrics"""
    if df.empty:
        return {}
    
    # Filter trades only (assuming 'T' action means trade)
    trades = df[df['action'] == 'T'].copy() if 'action' in df.columns else df.copy()
    
    if trades.empty:
        return {}
    
    metrics = {}
    
    # Basic metrics
    metrics['total_trades'] = len(trades)
    metrics['total_volume'] = trades['size'].sum()
    metrics['total_notional'] = trades['notional'].sum()
    metrics['avg_trade_size'] = trades['size'].mean()
    metrics['vwap'] = (trades['price'] * trades['size']).sum() / trades['size'].sum()
    
    # Price metrics
    metrics['price_min'] = trades['price'].min()
    metrics['price_max'] = trades['price'].max()
    metrics['price_range'] = metrics['price_max'] - metrics['price_min']
    metrics['price_std'] = trades['price'].std()
    
    # Time-based metrics
    time_span = (trades['ts_event'].max() - trades['ts_event'].min()).total_seconds()
    metrics['time_span_seconds'] = time_span
    metrics['trades_per_second'] = len(trades) / max(time_span, 1)
    
    # Price impact and volatility
    if len(trades) > 1:
        trades_sorted = trades.sort_values('ts_event')
        price_changes = trades_sorted['price'].diff().dropna()
        metrics['price_volatility'] = price_changes.std()
        metrics['max_price_move'] = abs(price_changes).max()
    
    # Order flow imbalance (if side data available)
    if 'side' in trades.columns:
        buy_volume = trades[trades['side'] == 'B']['size'].sum() if 'B' in trades['side'].values else 0
        sell_volume = trades[trades['side'] == 'S']['size'].sum() if 'S' in trades['side'].values else 0
        total_directional = buy_volume + sell_volume
        metrics['buy_volume'] = buy_volume
        metrics['sell_volume'] = sell_volume
        metrics['order_flow_imbalance'] = (buy_volume - sell_volume) / max(total_directional, 1)
    
    return metrics

def calculate_intraday_patterns(df):
    """Calculate intraday trading patterns"""
    if df.empty:
        return pd.DataFrame()
    
    trades = df[df['action'] == 'T'].copy() if 'action' in df.columns else df.copy()
    
    if trades.empty:
        return pd.DataFrame()
    
    # Group by hour
    hourly_stats = trades.groupby('hour').agg({
        'price': ['mean', 'std', 'min', 'max', 'count'],
        'size': ['sum', 'mean'],
        'notional': 'sum'
    }).round(4)
    
    hourly_stats.columns = ['price_mean', 'price_std', 'price_min', 'price_max', 'trade_count',
                           'volume_sum', 'avg_trade_size', 'notional_sum']
    
    return hourly_stats.reset_index()

def calculate_order_book_analytics(df):
    """Calculate order book analytics from quote data"""
    if df.empty:
        return pd.DataFrame()
    
    # Filter quotes (non-trade actions)
    quotes = df[df['action'] != 'T'].copy() if 'action' in df.columns else pd.DataFrame()
    
    if quotes.empty:
        return pd.DataFrame()
    
    # Group by timestamp and calculate spread metrics
    quote_analytics = quotes.groupby('second').agg({
        'price': ['min', 'max', 'count'],
        'size': ['sum', 'mean']
    }).round(4)
    
    quote_analytics.columns = ['bid_price', 'ask_price', 'quote_count', 'total_depth', 'avg_depth']
    quote_analytics['spread'] = quote_analytics['ask_price'] - quote_analytics['bid_price']
    quote_analytics['mid_price'] = (quote_analytics['ask_price'] + quote_analytics['bid_price']) / 2
    
    return quote_analytics.reset_index()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Tape Analysis", page_icon="📊", layout="wide")

# Custom CSS for enhanced styling
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.stMetric > div > div > div > div {
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# --- PAGE CONTENT ---
st.title("📊 High-Frequency Tape Analysis")
st.markdown("---")

st.markdown("""
**Advanced quantitative analysis of high-frequency Natural Gas futures tape data.**

Real-time market microstructure analytics including order flow, price discovery, and trading patterns.
""")

# Sidebar controls
with st.sidebar:
    st.title("🧭 Navigation")
    
    # Navigation buttons
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("streamlit_app.py")
    
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
    st.info("📍 **Current Page:** Tape Analysis")
    
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
    
    # Analysis type
    analysis_type = st.selectbox(
        "Analysis Type:",
        ["Market Microstructure", "Order Flow Analysis", "Intraday Patterns", "Price Discovery", "Liquidity Analysis"],
        help="Choose the type of quantitative analysis"
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

# Main content area
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
        
        st.subheader(f"📈 {contract_symbol} - Real-Time Analytics")
        
        # Key metrics dashboard
        col1, col2, col3, col4, col5 = st.columns(5)
        
        latest_price = df['price'].iloc[-1] if not df.empty else 0
        latest_time = df['ts_event'].iloc[-1] if not df.empty else None
        total_records = len(df)
        price_range = df['price'].max() - df['price'].min()
        avg_size = df['size'].mean()
        
        with col1:
            st.metric("Latest Price", f"${latest_price:.3f}")
        with col2:
            st.metric("Records", f"{total_records:,}")
        with col3:
            st.metric("Price Range", f"${price_range:.3f}")
        with col4:
            st.metric("Avg Size", f"{avg_size:,.0f}")
        with col5:
            if latest_time:
                time_ago = (datetime.now() - latest_time.replace(tzinfo=None)).total_seconds()
                st.metric("Last Update", f"{time_ago:.0f}s ago")
        
        if expiration:
            days_to_expiry = (expiration.date() - datetime.now().date()).days
            st.info(f"⏰ **Contract Expiration:** {expiration.date()} ({days_to_expiry} days remaining)")
        
        if analysis_type == "Market Microstructure":
            
            # Calculate microstructure metrics
            metrics = calculate_market_microstructure_metrics(df)
            
            if metrics:
                st.subheader("🔬 Market Microstructure Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Trades", f"{metrics.get('total_trades', 0):,}")
                    st.metric("VWAP", f"${metrics.get('vwap', 0):.3f}")
                
                with col2:
                    st.metric("Total Volume", f"{metrics.get('total_volume', 0):,.0f}")
                    st.metric("Avg Trade Size", f"{metrics.get('avg_trade_size', 0):.0f}")
                
                with col3:
                    st.metric("Price Volatility", f"{metrics.get('price_volatility', 0):.4f}")
                    st.metric("Max Price Move", f"${metrics.get('max_price_move', 0):.3f}")
                
                with col4:
                    st.metric("Trades/Second", f"{metrics.get('trades_per_second', 0):.2f}")
                    if 'order_flow_imbalance' in metrics:
                        imbalance = metrics['order_flow_imbalance']
                        st.metric("Order Flow Imbalance", f"{imbalance:.2%}")
            
            # Price and volume time series
            trades = df[df['action'] == 'T'].copy() if 'action' in df.columns else df.copy()
            
            if not trades.empty:
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=['Price Action', 'Trade Size', 'Cumulative Volume'],
                    vertical_spacing=0.08,
                    row_heights=[0.5, 0.25, 0.25]
                )
                
                # Price chart
                fig.add_trace(
                    go.Scatter(x=trades['ts_event'], y=trades['price'],
                              mode='lines', name='Price', line=dict(color='#ff6b35', width=1)),
                    row=1, col=1
                )
                
                # Add VWAP line
                if 'vwap' in metrics:
                    fig.add_hline(y=metrics['vwap'], line_dash="dash", line_color="blue",
                                annotation_text="VWAP", row=1, col=1)
                
                # Trade size
                colors = ['red' if side == 'S' else 'green' for side in trades.get('side', ['gray'] * len(trades))]
                fig.add_trace(
                    go.Scatter(x=trades['ts_event'], y=trades['size'],
                              mode='markers', name='Trade Size',
                              marker=dict(color=colors, size=4)),
                    row=2, col=1
                )
                
                # Cumulative volume
                cumulative_volume = trades['size'].cumsum()
                fig.add_trace(
                    go.Scatter(x=trades['ts_event'], y=cumulative_volume,
                              mode='lines', name='Cumulative Volume',
                              line=dict(color='purple', width=2)),
                    row=3, col=1
                )
                
                fig.update_layout(height=700, showlegend=False)
                fig.update_xaxes(title_text="Time", row=3, col=1)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Size", row=2, col=1)
                fig.update_yaxes(title_text="Volume", row=3, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Order Flow Analysis":
            
            if 'side' in df.columns:
                st.subheader("💹 Order Flow Analysis")
                
                trades = df[df['action'] == 'T'].copy() if 'action' in df.columns else df.copy()
                
                if not trades.empty:
                    # Order flow by side
                    buy_trades = trades[trades['side'] == 'B']
                    sell_trades = trades[trades['side'] == 'S']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        buy_vol = buy_trades['size'].sum() if not buy_trades.empty else 0
                        st.metric("Buy Volume", f"{buy_vol:,.0f}", delta=None)
                    
                    with col2:
                        sell_vol = sell_trades['size'].sum() if not sell_trades.empty else 0
                        st.metric("Sell Volume", f"{sell_vol:,.0f}", delta=None)
                    
                    with col3:
                        net_flow = buy_vol - sell_vol
                        st.metric("Net Flow", f"{net_flow:,.0f}", 
                                delta=f"{net_flow/max(buy_vol + sell_vol, 1):.1%}")
                    
                    # Order flow over time
                    trades['cumulative_buy'] = trades[trades['side'] == 'B']['size'].cumsum()
                    trades['cumulative_sell'] = trades[trades['side'] == 'S']['size'].cumsum()
                    
                    fig = go.Figure()
                    
                    if not buy_trades.empty:
                        fig.add_trace(go.Scatter(
                            x=buy_trades['ts_event'],
                            y=buy_trades['size'].cumsum(),
                            mode='lines',
                            name='Cumulative Buy Volume',
                            line=dict(color='green', width=2)
                        ))
                    
                    if not sell_trades.empty:
                        fig.add_trace(go.Scatter(
                            x=sell_trades['ts_event'],
                            y=sell_trades['size'].cumsum(),
                            mode='lines',
                            name='Cumulative Sell Volume',
                            line=dict(color='red', width=2)
                        ))
                    
                    fig.update_layout(
                        title="Cumulative Order Flow",
                        xaxis_title="Time",
                        yaxis_title="Cumulative Volume",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Side information not available for order flow analysis")
        
        elif analysis_type == "Intraday Patterns":
            
            st.subheader("🕐 Intraday Trading Patterns")
            
            patterns = calculate_intraday_patterns(df)
            
            if not patterns.empty:
                # Hourly volume and price patterns
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Hourly Volume', 'Hourly Price Range', 'Trade Count', 'Average Trade Size'],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                hours = patterns['hour'].dt.hour
                
                # Volume
                fig.add_trace(
                    go.Bar(x=hours, y=patterns['volume_sum'], name='Volume'),
                    row=1, col=1
                )
                
                # Price range
                fig.add_trace(
                    go.Scatter(x=hours, y=patterns['price_max'], 
                              mode='lines+markers', name='High', line=dict(color='green')),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(x=hours, y=patterns['price_min'], 
                              mode='lines+markers', name='Low', line=dict(color='red')),
                    row=1, col=2
                )
                
                # Trade count
                fig.add_trace(
                    go.Bar(x=hours, y=patterns['trade_count'], name='Trades'),
                    row=2, col=1
                )
                
                # Average trade size
                fig.add_trace(
                    go.Scatter(x=hours, y=patterns['avg_trade_size'], 
                              mode='lines+markers', name='Avg Size'),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display patterns table
                st.subheader("📊 Hourly Statistics")
                patterns_display = patterns.copy()
                patterns_display['hour'] = patterns_display['hour'].dt.strftime('%H:00')
                st.dataframe(patterns_display, use_container_width=True)
        
        elif analysis_type == "Price Discovery":
            
            st.subheader("🎯 Price Discovery Analysis")
            
            trades = df[df['action'] == 'T'].copy() if 'action' in df.columns else df.copy()
            
            if not trades.empty and len(trades) > 10:
                # Price impact analysis
                trades_sorted = trades.sort_values('ts_event')
                trades_sorted['price_change'] = trades_sorted['price'].diff()
                trades_sorted['volume_bucket'] = pd.qcut(trades_sorted['size'], 
                                                        q=5, labels=['XS', 'S', 'M', 'L', 'XL'])
                
                # Price impact by volume bucket
                impact_analysis = trades_sorted.groupby('volume_bucket').agg({
                    'price_change': ['mean', 'std', 'count'],
                    'size': ['mean', 'sum']
                }).round(4)
                
                impact_analysis.columns = ['avg_price_impact', 'price_impact_std', 'trade_count',
                                         'avg_size', 'total_volume']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Price impact chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=impact_analysis.index,
                        y=impact_analysis['avg_price_impact'],
                        name='Avg Price Impact',
                        marker_color='orange'
                    ))
                    fig.update_layout(
                        title="Price Impact by Trade Size",
                        xaxis_title="Volume Bucket",
                        yaxis_title="Average Price Change"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Volume distribution
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=impact_analysis.index,
                        y=impact_analysis['total_volume'],
                        name='Total Volume',
                        marker_color='blue'
                    ))
                    fig2.update_layout(
                        title="Volume Distribution by Bucket",
                        xaxis_title="Volume Bucket",
                        yaxis_title="Total Volume"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.dataframe(impact_analysis, use_container_width=True)
        
        elif analysis_type == "Liquidity Analysis":
            
            st.subheader("💧 Liquidity Analysis")
            
            # Calculate liquidity metrics
            trades = df[df['action'] == 'T'].copy() if 'action' in df.columns else df.copy()
            
            if not trades.empty:
                # Time-based liquidity (trades per minute)
                minute_stats = trades.groupby('minute').agg({
                    'size': ['sum', 'count'],
                    'price': ['std', 'min', 'max']
                }).round(4)
                
                minute_stats.columns = ['volume', 'trade_count', 'price_volatility', 'price_min', 'price_max']
                minute_stats['liquidity_score'] = minute_stats['volume'] / (minute_stats['price_volatility'] + 0.001)
                
                # Liquidity visualization
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=['Trading Volume per Minute', 'Liquidity Score (Volume/Volatility)'],
                    vertical_spacing=0.1
                )
                
                fig.add_trace(
                    go.Bar(x=minute_stats.index, y=minute_stats['volume'], name='Volume'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=minute_stats.index, y=minute_stats['liquidity_score'], 
                              mode='lines', name='Liquidity Score', line=dict(color='purple')),
                    row=2, col=1
                )
                
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Liquidity summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_liquidity = minute_stats['liquidity_score'].mean()
                    st.metric("Avg Liquidity Score", f"{avg_liquidity:.2f}")
                
                with col2:
                    most_liquid_minute = minute_stats['liquidity_score'].idxmax()
                    st.metric("Most Liquid Time", most_liquid_minute.strftime('%H:%M'))
                
                with col3:
                    total_active_minutes = len(minute_stats)
                    st.metric("Active Minutes", total_active_minutes)

        # Raw data section
        with st.expander("📋 Raw Tape Data Sample"):
            # Show last 100 records
            display_df = df.tail(100).copy()
            display_df['ts_event'] = display_df['ts_event'].dt.strftime('%H:%M:%S.%f').str[:-3]
            
            st.dataframe(
                display_df[['ts_event', 'price', 'size', 'action', 'side', 'raw_symbol']],
                use_container_width=True
            )
            
            if st.button("📥 Download Full Dataset"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{contract_symbol}_tape_data.csv",
                    mime="text/csv"
                )

st.markdown("---")
st.markdown("**💾 Data Source:** High-Frequency Tape Data | **🔄 Updates:** Real-time")