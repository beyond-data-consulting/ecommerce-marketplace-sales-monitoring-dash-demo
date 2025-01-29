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
        'Jacket': {'price_mean': 120, 'price_std': 15, 'daily_orders': 20}
    }
    
    all_data = []
    
    # Generate data for each product
    for product_name, specs in products.items():
        print(product_name)
        # Calculate number of orders per 15 minutes
        orders_per_interval = max(1, int(specs['daily_orders'] * (15 / (24 * 60))))
        
        # Generate base data for this product
        base_df = pd.DataFrame({
            'timestamp': dates,
            'product_name': product_name
        })
        
        # Add daily trend component (slight increase during business hours)
        hour_effect = np.sin(2 * np.pi * base_df['timestamp'].dt.hour / 24) * 0.02
        
        # Add weekly trend (slightly higher prices on weekends)
        weekend_effect = (base_df['timestamp'].dt.dayofweek >= 5).astype(float) * 0.015
        
        # Add monthly trend (slight increase towards end of month)
        monthly_effect = (base_df['timestamp'].dt.day / 31) * 0.01
        
        # Combine all effects and add random noise
        total_effect = (1 + hour_effect + weekend_effect + monthly_effect)
        random_noise = np.random.normal(0, 0.005, len(base_df))  # 0.5% random noise
        
        # Calculate final prices
        base_df['amount_usd'] = (
            specs['price_mean'] * 
            total_effect * 
            (1 + random_noise) + 
            np.random.normal(0, specs['price_std'] * 0.1, len(base_df))
        )
        
        # Clip prices to reasonable range (allowing more variation than before)
        base_df['amount_usd'] = base_df['amount_usd'].clip(
            specs['price_mean'] * 0.85,  # Allow 15% decrease
            specs['price_mean'] * 1.15   # Allow 15% increase
        )
        
        # Add is_express column (20% chance of being express)
        base_df['is_express'] = np.random.choice(
            [True, False], 
            size=len(base_df), 
            p=[0.2, 0.8]
        )
        
        # Sample rows based on expected order volume
        product_data = base_df.sample(n=orders_per_interval * len(dates), replace=True)
        all_data.append(product_data)

    df = pd.concat(all_data, ignore_index=True)
    print(df['product_name'].value_counts())
    
    # Create duplicate orders for seasonal patterns
    extra_orders = []
    
    # Seasonal trends
    # Hoodies: September through January
    hoodie_mask = (df['product_name'] == 'Hoodie') & (
        ((df['timestamp'].dt.month >= 9) & (df['timestamp'].dt.month <= 12)) |
        (df['timestamp'].dt.month == 1)
    )
    hoodie_seasonal = df[hoodie_mask].copy()
    for month in [9, 10, 11, 12, 1]:
        month_mask = hoodie_seasonal['timestamp'].dt.month == month
        # Peak in November/December (100% more orders), ramping up from September (20%) and down in January (20%)
        if month in [11, 12]:
            duplicate_factor = 1.0  # 100% more orders
        elif month in [10]:
            duplicate_factor = 0.6  # 60% more orders
        else:  # September and January
            duplicate_factor = 0.2  # 20% more orders
        
        n_duplicates = int(len(hoodie_seasonal[month_mask]) * duplicate_factor)
        if n_duplicates > 0:
            duplicates = hoodie_seasonal[month_mask].sample(n=n_duplicates, replace=True)
            duplicates['timestamp'] += pd.Timedelta(minutes=np.random.randint(1, 10))
            extra_orders.append(duplicates)

    # T-Shirts: March through June
    tshirt_mask = (df['product_name'] == 'T-Shirt') & (
        (df['timestamp'].dt.month >= 3) & (df['timestamp'].dt.month <= 6)
    )
    tshirt_seasonal = df[tshirt_mask].copy()
    for month in [3, 4, 5, 6]:
        month_mask = tshirt_seasonal['timestamp'].dt.month == month
        # Peak in May (80% more orders), ramping up from March and down in June
        if month == 5:
            duplicate_factor = 0.3  # 80% more orders
        elif month == 4:
            duplicate_factor = 0.2  # 60% more orders
        else:  # March and June
            duplicate_factor = 0.15  # 30% more orders
        
        n_duplicates = int(len(tshirt_seasonal[month_mask]) * duplicate_factor)
        if n_duplicates > 0:
            duplicates = tshirt_seasonal[month_mask].sample(n=n_duplicates, replace=True)
            duplicates['timestamp'] += pd.Timedelta(minutes=np.random.randint(1, 10))
            extra_orders.append(duplicates)

    # Jackets: August through February
    jacket_mask = (df['product_name'] == 'Jacket') & (
        ((df['timestamp'].dt.month >= 8) & (df['timestamp'].dt.month <= 12)) |
        (df['timestamp'].dt.month <= 2)
    )
    jacket_seasonal = df[jacket_mask].copy()
    for month in [8, 9, 10, 11, 12, 1, 2]:
        month_mask = jacket_seasonal['timestamp'].dt.month == month
        # Peak in December/January (120% more orders), ramping up from August (30%) and down in February (30%)
        if month in [12, 1]:
            duplicate_factor = 1.2  # 120% more orders
        elif month in [11, 2]:
            duplicate_factor = 0.8  # 80% more orders
        elif month in [10]:
            duplicate_factor = 0.6  # 60% more orders
        else:  # August, September, February
            duplicate_factor = 0.3  # 30% more orders
        
        n_duplicates = int(len(jacket_seasonal[month_mask]) * duplicate_factor)
        if n_duplicates > 0:
            duplicates = jacket_seasonal[month_mask].sample(n=n_duplicates, replace=True)
            duplicates['timestamp'] += pd.Timedelta(minutes=np.random.randint(1, 10))
            extra_orders.append(duplicates)

    # Jeans: August through May
    jeans_mask = (df['product_name'] == 'Jeans') & (
        ((df['timestamp'].dt.month >= 8) & (df['timestamp'].dt.month <= 12)) |
        (df['timestamp'].dt.month <= 5)
    )
    jeans_seasonal = df[jeans_mask].copy()
    for month in [8, 9, 10, 11, 12, 1, 2, 3, 4, 5]:
        month_mask = jeans_seasonal['timestamp'].dt.month == month
        # Peak in October/November (90% more orders), with gradual ramp up and down
        if month in [10, 11]:
            duplicate_factor = 0.9  # 90% more orders
        elif month in [9, 12]:
            duplicate_factor = 0.6  # 60% more orders
        elif month in [8, 1]:
            duplicate_factor = 0.4  # 40% more orders
        else:  # February through May
            duplicate_factor = 0.2  # 20% more orders
        
        n_duplicates = int(len(jeans_seasonal[month_mask]) * duplicate_factor)
        if n_duplicates > 0:
            duplicates = jeans_seasonal[month_mask].sample(n=n_duplicates, replace=True)
            duplicates['timestamp'] += pd.Timedelta(minutes=np.random.randint(1, 10))
            extra_orders.append(duplicates)

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
    print(df['product_name'].value_counts())
    
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

# Create three columns for charts
col1, col2, col3 = st.columns([4, 4, 2])  # Adjust width ratios

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

with col3:
    st.subheader("Past 24 Hours")
    
    # Calculate last 24 hours statistics using the max timestamp in our data
    max_timestamp = filtered_df['timestamp'].max()
    last_24h = max_timestamp - timedelta(days=1)
    last_24h_data = filtered_df[filtered_df['timestamp'] >= last_24h]
    
    # Prepare summary table data
    summary_stats = last_24h_data.groupby('product_name').agg({
        'amount_usd': 'mean',
        'product_name': 'size',
        'is_express': 'sum'  # Sum of True values gives us count of express orders
    }).rename(columns={
        'amount_usd': 'avg_price',
        'product_name': 'num_orders',
        'is_express': 'express_orders'
    }).reset_index()
    
    # Sort by number of orders descending
    summary_stats = summary_stats.sort_values('num_orders', ascending=False)
    
    # Format the numbers
    summary_stats['avg_price'] = summary_stats['avg_price'].round(2).apply(lambda x: f"${x}")
    summary_stats['num_orders'] = summary_stats['num_orders'].apply(lambda x: f"{x:,}")
    summary_stats['express_orders'] = summary_stats['express_orders'].astype(int).apply(lambda x: f"{x:,}")
    
    # Display as a clean table
    st.dataframe(
        summary_stats,
        column_config={
            "product_name": "Product",
            "avg_price": "Avg Price",
            "num_orders": "Orders",
            "express_orders": "Express Orders"
        },
        hide_index=True
    )

# Requirements to run:
# pip install streamlit pandas numpy plotly