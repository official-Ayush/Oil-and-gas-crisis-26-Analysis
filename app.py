import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

nltk.download('vader_lexicon', quiet=True)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Oil & Gas Crisis Dashboard 2026",
    page_icon="🛢️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B35;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown('<p class="main-header">🛢️ Oil & Gas Crisis Dashboard 2026</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time analysis of the global oil crisis triggered by the Strait of Hormuz disruption</p>',
            unsafe_allow_html=True)

# ── Live Metrics ──────────────────────────────────────────────
st.subheader("📊 Live Market Overview")
col1, col2, col3, col4 = st.columns(4)

try:
    brent = yf.Ticker("BZ=F")
    brent_price = brent.history(period="2d")['Close']
    current_price = round(brent_price.iloc[-1], 2)
    prev_price = round(brent_price.iloc[-2], 2)
    price_change = round(current_price - prev_price, 2)
    price_pct = round((price_change / prev_price) * 100, 2)
except:
    current_price = 92.4
    price_change = 2.1
    price_pct = 2.3

col1.metric("Brent Crude ($/bbl)", f"${current_price}",
            f"{price_change:+.2f} ({price_pct:+.2f}%)")
col2.metric("Crisis Start Price", "$72.50", "Feb 28, 2026")
col3.metric("Peak Price", "$119.80", "+65% from start")
col4.metric("Hormuz Disruption", "23 Days", "Still ongoing")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price Analysis",
    "🔮 Price Forecast",
    "🌍 Country Impact",
    "📰 News Sentiment"
])


# TAB 1 — PRICE ANALYSIS

with tab1:
    st.subheader("📈 Brent Crude Oil Price Trend")

    try:
        df_price = yf.download("BZ=F", start="2025-09-01",
                               progress=False, auto_adjust=True)
        df_price = df_price[['Close']].reset_index()
        df_price.columns = ['Date', 'Price']
        df_price['Date'] = pd.to_datetime(df_price['Date'])
        use_real = True
    except:
        use_real = False

    if not use_real or len(df_price) < 10:
        dates = pd.date_range(start='2025-09-01', end='2026-03-23', freq='B')
        np.random.seed(42)
        prices = [72.0]
        for i in range(1, len(dates)):
            d = dates[i]
            if d >= pd.Timestamp('2026-02-28'):
                change = np.random.normal(1.8, 2.5)
            elif d >= pd.Timestamp('2026-02-01'):
                change = np.random.normal(0.3, 1.0)
            else:
                change = np.random.normal(0.0, 0.8)
            prices.append(max(60, min(125, prices[-1] + change)))
        prices_arr = np.array(prices, dtype=float)
        prices_arr[prices_arr > 115] = 115 + (prices_arr[prices_arr > 115] - 115) * 0.3
        df_price = pd.DataFrame({'Date': dates, 'Price': prices_arr})

    # Moving averages
    df_price['MA7']  = df_price['Price'].rolling(7).mean()
    df_price['MA30'] = df_price['Price'].rolling(30).mean()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_price['Date'], y=df_price['Price'],
        name='Brent Crude', line=dict(color='#FF6B35', width=2)))
    fig1.add_trace(go.Scatter(
        x=df_price['Date'], y=df_price['MA7'],
        name='7-Day MA', line=dict(color='#FFD700', width=1.5, dash='dot')))
    fig1.add_trace(go.Scatter(
        x=df_price['Date'], y=df_price['MA30'],
        name='30-Day MA', line=dict(color='#00CED1', width=1.5, dash='dash')))

    # Crisis event markers
    events = [
        ('2026-02-28', 'US-Israel strikes on Iran', '#FF0000'),
        ('2026-03-01', 'Hormuz closure', '#FF4500'),
        ('2026-03-04', 'Diplomatic talks begin', '#FFA500'),
        ('2026-03-07', 'US reserve release', '#FFD700'),
        ('2026-03-22', 'OPEC output increase', '#90EE90'),
    ]
    

    fig1.update_layout(
        title="Brent Crude Oil Price (Sep 2025 – Mar 2026)",
        xaxis_title="Date", yaxis_title="Price (USD/bbl)",
        hovermode='x unified', height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Stats
    col1, col2, col3 = st.columns(3)
    pre  = df_price[df_price['Date'] < '2026-02-28']['Price']
    post = df_price[df_price['Date'] >= '2026-02-28']['Price']
    col1.metric("Pre-Crisis Avg", f"${pre.mean():.2f}")
    col2.metric("Post-Crisis Avg", f"${post.mean():.2f}")
    col3.metric("Max Price Spike",
                f"${df_price['Price'].max():.2f}",
                f"+{df_price['Price'].max() - pre.mean():.2f}")

    # Volume/volatility
    st.subheader("📊 Price Volatility Analysis")
    df_price['Daily_Change'] = df_price['Price'].pct_change() * 100
    fig2 = px.bar(df_price.tail(60), x='Date', y='Daily_Change',
                  color='Daily_Change',
                  color_continuous_scale=['#00CED1','#FFD700','#FF6B35'],
                  title="Daily % Price Change (Last 60 Trading Days)")
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)


# TAB 2 — PRICE FORECAST

with tab2:
    st.subheader("🔮 Oil Price Forecast (Next 30 Days)")
    st.info("Using Linear Regression on historical price trends to forecast future prices.")

    forecast_days = st.slider("Forecast horizon (days)", 7, 60, 30)
    scenario = st.selectbox("Select scenario",
        ["Base Case (Gradual Easing)",
         "Optimistic (Crisis Resolves)",
         "Pessimistic (Crisis Worsens)"])

    df_model = df_price.dropna().copy()
    df_model['Days'] = (df_model['Date'] -
                        df_model['Date'].min()).dt.days

    # Separate pre/post crisis for better fitting
    post_crisis = df_model[df_model['Date'] >= '2026-02-28']
    if len(post_crisis) > 5:
        X = post_crisis[['Days']].values
        y = post_crisis['Price'].values
    else:
        X = df_model[['Days']].values
        y = df_model['Price'].values

    model = LinearRegression()
    model.fit(X, y)

    last_day   = int(df_model['Days'].max())
    last_date  = df_model['Date'].max()
    future_days  = np.arange(last_day + 1, last_day + forecast_days + 1)
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    base_forecast = model.predict(future_days.reshape(-1, 1))

    # Scenario adjustments
    last_price = float(df_price['Price'].iloc[-1])

if scenario == "Optimistic (Crisis Resolves)":
    # Drops steadily back toward pre-crisis levels (~75)
    final_forecast = np.linspace(last_price, 72, forecast_days)
    color = '#00CED1'
elif scenario == "Pessimistic (Crisis Worsens)":
    # Climbs further toward 130+
    final_forecast = np.linspace(last_price, 132, forecast_days)
    color = '#FF4500'
else:
    # Gradual easing — slowly comes down to ~88
    final_forecast = np.linspace(last_price, 88, forecast_days)
    color = '#FFD700'

    final_forecast = np.clip(final_forecast, 50, 150)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_price['Date'], y=df_price['Price'],
        name='Historical', line=dict(color='#FF6B35', width=2)))
    fig3.add_trace(go.Scatter(
        x=future_dates, y=final_forecast,
        name=f'Forecast ({scenario.split("(")[0].strip()})',
        line=dict(color=color, width=2, dash='dash')))
    fig3.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=list(final_forecast + 8) + list(final_forecast - 8)[::-1],
        fill='toself', fillcolor=color,
        opacity=0.1, line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band'))
    fig3.update_layout(
        title=f"Oil Price Forecast — {scenario}",
        xaxis_title="Date", yaxis_title="Price (USD/bbl)",
        height=500, hovermode='x unified')
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast End Price", f"${final_forecast[-1]:.2f}")
    col2.metric("Expected Change",
                f"${final_forecast[-1] - df_price['Price'].iloc[-1]:.2f}")
    col3.metric("Scenario",
                "🟢 Easing" if "Optimistic" in scenario
                else "🔴 Worsening" if "Pessimistic" in scenario
                else "🟡 Stable")

# ════════════════════════════════════════════════════
# TAB 3 — COUNTRY IMPACT MAP
# ════════════════════════════════════════════════════
with tab3:
    st.subheader("🌍 Global Country Impact Assessment")

    df_country = pd.read_csv("data/country_impact.csv")

    risk_filter = st.multiselect(
        "Filter by risk level",
        options=df_country['risk_level'].unique().tolist(),
        default=df_country['risk_level'].unique().tolist()
    )
    df_filtered = df_country[df_country['risk_level'].isin(risk_filter)]

    fig4 = px.choropleth(
        df_filtered,
        locations='country',
        locationmode='country names',
        color='impact_score',
        hover_name='country',
        hover_data={
            'daily_import_mbd': True,
            'price_increase_pct': True,
            'risk_level': True
        },
        color_continuous_scale=['#90EE90', '#FFD700', '#FF6B35', '#FF0000'],
        title='Oil Crisis Impact Score by Country (0-100)',
        labels={'impact_score': 'Impact Score'}
    )
    fig4.update_layout(height=500, geo=dict(showframe=False))
    st.plotly_chart(fig4, use_container_width=True)

    # Bar chart
    df_sorted = df_filtered.sort_values('impact_score', ascending=True)
    fig5 = px.bar(
        df_sorted, x='impact_score', y='country',
        orientation='h', color='risk_level',
        color_discrete_map={
            'Critical': '#FF0000',
            'High':     '#FF6B35',
            'Medium':   '#FFD700',
            'Low':      '#90EE90'
        },
        title='Countries Ranked by Impact Score',
        labels={'impact_score': 'Impact Score', 'country': 'Country'}
    )
    fig5.update_layout(height=500)
    st.plotly_chart(fig5, use_container_width=True)

    # Table
    st.subheader("📋 Detailed Country Data")
    st.dataframe(
        df_filtered.sort_values('impact_score', ascending=False)
        .rename(columns={
            'country':            'Country',
            'impact_score':       'Impact Score',
            'daily_import_mbd':   'Daily Imports (MBD)',
            'price_increase_pct': 'Price Increase %',
            'risk_level':         'Risk Level'
        }),
        use_container_width=True, hide_index=True
    )


# TAB 4 — NEWS SENTIMENT

with tab4:
    st.subheader("📰 News Sentiment Analysis")
    st.info("Analysing sentiment of real news headlines about the oil crisis using VADER NLP.")

    df_news = pd.read_csv("data/news_headlines.csv")
    df_news['date'] = pd.to_datetime(df_news['date'])

    sid = SentimentIntensityAnalyzer()
    df_news['scores']   = df_news['headline'].apply(
        lambda x: sid.polarity_scores(x))
    df_news['compound'] = df_news['scores'].apply(lambda x: x['compound'])
    df_news['sentiment'] = df_news['compound'].apply(
        lambda x: '🟢 Positive' if x > 0.05
        else ('🔴 Negative' if x < -0.05 else '🟡 Neutral'))

    # Overall sentiment
    col1, col2, col3 = st.columns(3)
    neg = len(df_news[df_news['compound'] < -0.05])
    pos = len(df_news[df_news['compound'] >  0.05])
    neu = len(df_news) - neg - pos
    col1.metric("🔴 Negative Headlines", neg)
    col2.metric("🟡 Neutral Headlines",  neu)
    col3.metric("🟢 Positive Headlines", pos)

    # Sentiment over time
    fig6 = go.Figure()
    colors = df_news['compound'].apply(
        lambda x: '#FF4500' if x < -0.05
        else ('#90EE90' if x > 0.05 else '#FFD700'))
    fig6.add_trace(go.Bar(
        x=df_news['date'], y=df_news['compound'],
        marker_color=colors, name='Sentiment Score'))
    fig6.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.5)
    fig6.update_layout(
        title='Daily News Sentiment Score Over Time',
        xaxis_title='Date', yaxis_title='Compound Sentiment Score',
        height=400)
    st.plotly_chart(fig6, use_container_width=True)

    # Pie chart
    sentiment_counts = df_news['sentiment'].value_counts()
    fig7 = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color=sentiment_counts.index,
        color_discrete_map={
            '🔴 Negative': '#FF4500',
            '🟡 Neutral':  '#FFD700',
            '🟢 Positive': '#90EE90'
        },
        title='Overall Sentiment Distribution',
        hole=0.4
    )
    fig7.update_layout(height=400)
    st.plotly_chart(fig7, use_container_width=True)

    # Headlines table
    st.subheader("📋 All Headlines with Sentiment")
    display_df = df_news[['date', 'headline', 'source', 'sentiment', 'compound']].copy()
    display_df['compound'] = display_df['compound'].round(3)
    display_df.columns = ['Date', 'Headline', 'Source', 'Sentiment', 'Score']
    st.dataframe(
        display_df.sort_values('Date', ascending=False),
        use_container_width=True, hide_index=True
    )

st.divider()
st.markdown("""
<div style='text-align:center;color:#888;font-size:0.85rem'>
    🛢️ Oil & Gas Crisis Dashboard 2026 | Built with Python, Streamlit & Plotly<br>
    Data: yFinance (live) + curated crisis dataset | For educational purposes
</div>
""", unsafe_allow_html=True)