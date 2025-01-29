import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Sales Dashboard", layout="wide")

# Add this near the top of the file, after the imports
@st.cache_data
def prepare_sales_volume_data(df):
    return df.groupby(
        [pd.Grouper(key='timestamp', freq='W-MON'), 'product_name']
    ).size().reset_index(name='count')

# Generate dummy data
def generate_dummy_data(start_date='2023-01-01', end_date='2023-12-31'):
    # Create date range (every 15 minutes instead of hourly)
    dates = pd.date_range(start=start_date, end=end_date, freq='15min')

    # Define products with their characteristics
    products = {
        'Hoodie': {'price_mean': 55, 'price_std': 8, 'daily_orders': 100},
        'T-Shirt': {'price_mean': 25, 'price_std': 5, 'daily_orders': 200},
        'Jeans': {'price_mean': 80, 'price_std': 12, 'daily_orders': 60},
        'Socks': {'price_mean': 12, 'price_std': 2, 'daily_orders': 150},
        'Jacket': {'price_mean': 120, 'price_std': 15, 'daily_orders': 40}
    }
    
    all_data = []
    
    # Generate data for each product
    for product_name, specs in products.items():
        # Calculate number of orders per 15 minutes
        orders_per_interval = int(specs['daily_orders'] * (15 / (24 * 60)))
        
        # Generate base data for this product
        product_data = pd.DataFrame({
            'timestamp': dates,
            'product_name': product_name,
            'amount_usd': np.random.normal(
                specs['price_mean'], 
                specs['price_std'], 
                len(dates)
            ).clip(specs['price_mean'] * 0.7, specs['price_mean'] * 1.3)  # Clip prices to reasonable range
        })
        
        # Sample rows based on expected order volume
        product_data = product_data.sample(n=orders_per_interval * len(dates), replace=True)
        all_data.append(product_data)

    df = pd.concat(all_data, ignore_index=True)
    
    # Create duplicate orders for seasonal patterns
    extra_orders = []
    
    # Christmas build-up (December 1-25)
    christmas_mask = (df['timestamp'].dt.month == 12) & (df['timestamp'].dt.day <= 25)
    christmas_df = df[christmas_mask].copy()
    for day in range(1, 26):
        day_mask = christmas_df['timestamp'].dt.day == day
        duplicate_factor = 0.2 + (0.6 * (day - 1) / 24)  # 20% to 80% more orders
        n_duplicates = int(len(christmas_df[day_mask]) * duplicate_factor)
        if n_duplicates > 0:
            duplicates = christmas_df[day_mask].sample(n=n_duplicates, replace=True)
            duplicates['timestamp'] += pd.Timedelta(minutes=np.random.randint(1, 10))
            extra_orders.append(duplicates)

    # Black Friday build-up (November 20-27)
    bf_mask = (df['timestamp'].dt.month == 11) & (df['timestamp'].dt.day.between(20, 27))
    bf_df = df[bf_mask].copy()
    for day in range(20, 28):
        day_mask = bf_df['timestamp'].dt.day == day
        duplicate_factor = 0.2 + (0.3 * (day - 20) / 7)  # 20% to 50% more orders
        n_duplicates = int(len(bf_df[day_mask]) * duplicate_factor)
        if n_duplicates > 0:
            duplicates = bf_df[day_mask].sample(n=n_duplicates, replace=True)
            duplicates['timestamp'] += pd.Timedelta(minutes=np.random.randint(1, 10))
            extra_orders.append(duplicates)

    # Combine all data
    if extra_orders:
        df = pd.concat([df] + extra_orders, ignore_index=True)
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    return df

# Add this after the generate_dummy_data function definition
@st.cache_data
def get_cached_data():
    return generate_dummy_data()

# Then replace the data generation line
# Generate data
df = get_cached_data()  # Instead of df = generate_dummy_data()

# Dashboard title
st.title("📊 Sales Dashboard")
st.markdown("---")

# Remove sidebar filters section entirely
filtered_df = df.copy(deep=True)

# Create two columns for charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales Volume by Product")

    # Use cached data preparation
    daily_sales = prepare_sales_volume_data(df)

    # Create timeseries chart
    fig_timeseries = px.line(
        daily_sales,
        x='timestamp',
        y='count',
        color='product_name',
        title='Weekly Order Volume',
        labels={'timestamp': 'Week', 'count': 'Number of Orders', 'product_name': 'Product'}
    )
    st.plotly_chart(fig_timeseries, use_container_width=True)

with col2:
    st.subheader("Product Price Analysis")

    # Single select for candlestick
    selected_product = st.selectbox(
        "Select Product",
        options=sorted(df['product_name'].unique())
    )

    # Prepare data for candlestick
    daily_prices = df[df['product_name'] == selected_product].groupby(
        df['timestamp'].dt.date
    ).agg({
        'amount_usd': ['first', 'max', 'min', 'last']
    }).reset_index()

    daily_prices.columns = ['date', 'open', 'high', 'low', 'close']

    # Create candlestick chart
    fig_candlestick = go.Figure(data=[go.Candlestick(
        x=daily_prices['date'],
        open=daily_prices['open'],
        high=daily_prices['high'],
        low=daily_prices['low'],
        close=daily_prices['close']
    )])

    fig_candlestick.update_layout(
        title=f'Daily Price Analysis - {selected_product}',
        xaxis_title='Date',
        yaxis_title='Price (USD)'
    )

    st.plotly_chart(fig_candlestick, use_container_width=True)

# Add summary metrics
st.markdown("---")
st.subheader("Summary Metrics")

# Create metrics row
metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    total_sales = filtered_df['amount_usd'].sum()
    st.metric("Total Sales", f"${total_sales:,.2f}")

with metric2:
    avg_daily_sales = filtered_df.groupby(filtered_df['timestamp'].dt.date)['amount_usd'].sum().mean()
    st.metric("Avg Daily Sales", f"${avg_daily_sales:,.2f}")

with metric3:
    top_product = filtered_df.groupby('product_name')['amount_usd'].sum().idxmax()
    st.metric("Top Selling Product", f"Product {top_product}")

with metric4:
    total_transactions = len(filtered_df)
    st.metric("Total Transactions", f"{total_transactions:,}")

# Requirements to run:
# pip install streamlit pandas numpy plotly