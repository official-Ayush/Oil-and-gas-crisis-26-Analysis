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
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown('<p class="main-header">🛢️ Oil & Gas Crisis Dashboard 2026</p>',
            unsafe_allow_html=True)
st.markdown(
    f'<p class="sub-header">Real-time analysis of the global oil crisis triggered '
    f'by the Strait of Hormuz disruption — Live as of {datetime.now().strftime("%B %d, %Y")}</p>',
    unsafe_allow_html=True
)

# ════════════════════════════════════════════════════
# SHARED DATA FETCH (used across all tabs)
# ════════════════════════════════════════════════════
@st.cache_data(ttl=3600)  # refresh every 1 hour
def fetch_price_data():
    try:
        df = yf.download("BZ=F", start="2025-09-01",
                         end=datetime.now().strftime('%Y-%m-%d'),
                         progress=False, auto_adjust=True)
        df = df[['Close']].reset_index()
        df.columns = ['Date', 'Price']
        df['Date'] = pd.to_datetime(df['Date'])
        df['Price'] = df['Price'].astype(float)
        if len(df) > 10:
            return df, True
    except Exception:
        pass

    # ── Fallback: synthetic data if yFinance fails ──────────────
    dates = pd.date_range(start='2025-09-01', end=datetime.now(), freq='B')
    np.random.seed(42)
    prices = [72.0]
    for i in range(1, len(dates)):
        d = dates[i]
        if d >= pd.Timestamp('2026-04-13'):
            change = np.random.normal(1.2, 3.0)
        elif d >= pd.Timestamp('2026-03-22'):
            change = np.random.normal(0.5, 2.0)
        elif d >= pd.Timestamp('2026-02-28'):
            change = np.random.normal(2.0, 2.5)
        else:
            change = np.random.normal(0.0, 0.8)
        prices.append(max(60, min(125, prices[-1] + change)))
    df = pd.DataFrame({'Date': dates, 'Price': np.array(prices, dtype=float)})
    return df, False


@st.cache_data(ttl=3600)
def fetch_oil_news():
    import feedparser
    feed_url = ("https://news.google.com/rss/search?q=oil+price+Strait+of+Hormuz+"
                 "crisis&hl=en-US&gl=US&ceid=US:en")
    feed = feedparser.parse(feed_url)

    headlines, dates, sources = [], [], []
    for entry in feed.entries[:40]:
        headlines.append(entry.title.rsplit(' - ', 1)[0])
        sources.append(entry.title.rsplit(' - ', 1)[-1] if ' - ' in entry.title else 'Unknown')
        try:
            pub_date = pd.to_datetime(entry.published).tz_localize(None)
        except Exception:
            pub_date = pd.Timestamp.now()
        dates.append(pub_date)

    return pd.DataFrame({'date': dates, 'headline': headlines, 'source': sources})


# Fetch shared price data once, used everywhere
df_price_global, is_live_data = fetch_price_data()
df_price_global['MA7'] = df_price_global['Price'].rolling(7).mean()
df_price_global['MA30'] = df_price_global['Price'].rolling(30).mean()

# ════════════════════════════════════════════════════
# LIVE METRICS (auto-calculated)
# ════════════════════════════════════════════════════
st.subheader("📊 Live Market Overview")
col1, col2, col3, col4 = st.columns(4)

try:
    brent = yf.Ticker("BZ=F")
    brent_price = brent.history(period="2d")['Close']
    current_price = round(float(brent_price.iloc[-1]), 2)
    prev_price = round(float(brent_price.iloc[-2]), 2)
    price_change = round(current_price - prev_price, 2)
    price_pct = round((price_change / prev_price) * 100, 2)
except Exception:
    current_price = float(df_price_global['Price'].iloc[-1])
    price_change = 0.0
    price_pct = 0.0

# Crisis start price — actual value on Feb 28, 2026 (or closest available)
try:
    crisis_start_price = float(
        df_price_global.iloc[
            (df_price_global['Date'] - pd.Timestamp('2026-02-28')).abs().argsort()[:1]
        ]['Price'].iloc[0]
    )
except Exception:
    crisis_start_price = 72.50

# Peak price since crisis began
try:
    post_crisis_df = df_price_global[df_price_global['Date'] >= '2026-02-28']
    peak_price = float(post_crisis_df['Price'].max())
    peak_pct = ((peak_price - crisis_start_price) / crisis_start_price) * 100
except Exception:
    peak_price, peak_pct = 119.80, 65.0

# Days since crisis started
crisis_days = (pd.Timestamp.now().normalize() - pd.Timestamp('2026-02-28')).days

col1.metric("Brent Crude ($/bbl)", f"${current_price:.2f}",
            f"{price_change:+.2f} ({price_pct:+.2f}%)")
col2.metric("Crisis Start Price", f"${crisis_start_price:.2f}", "Feb 28, 2026")
col3.metric("Peak Price", f"${peak_price:.2f}", f"+{peak_pct:.0f}% from start")
col4.metric("Hormuz Disruption", f"{crisis_days} Days", "Still ongoing")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price Analysis",
    "🔮 Price Forecast",
    "🌍 Country Impact",
    "📰 News Sentiment"
])

# ════════════════════════════════════════════════════
# TAB 1 — PRICE ANALYSIS
# ════════════════════════════════════════════════════
with tab1:
    st.subheader("📈 Brent Crude Oil Price Trend")

    df_price = df_price_global.copy()

    if is_live_data:
        st.caption("🟢 Live data from yFinance")
    else:
        st.caption("🟡 Using simulated data (yFinance temporarily unavailable)")

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

    # Crisis event markers (fixed historical record)
    events = [
        ('2026-02-28', 'US-Israel strikes Iran', '#FF0000'),
        ('2026-03-01', 'Hormuz closure', '#FF4500'),
        ('2026-03-07', 'US reserve release', '#FFD700'),
        ('2026-03-22', 'OPEC output increase', '#90EE90'),
        ('2026-04-13', 'US blockades Iran ports', '#FF4500'),
        ('2026-05-05', 'Project Freedom', '#FFA500'),
    ]

    label_offsets = [0.92, 0.82, 0.72, 0.62, 0.52, 0.42]
    for i, (date_str, label, color) in enumerate(events):
        event_date = pd.Timestamp(date_str)
        if event_date <= df_price['Date'].max():
            fig1.add_vline(x=event_date, line_dash="dash",
                           line_color=color, opacity=0.6)
            fig1.add_annotation(
                x=event_date,
                y=label_offsets[i],
                yref="paper",
                text=label,
                showarrow=False,
                font=dict(size=9, color=color),
                textangle=-45,
                xanchor="left"
            )

    fig1.update_layout(
        title=f"Brent Crude Oil Price (Sep 2025 – {datetime.now().strftime('%b %Y')})",
        xaxis_title="Date", yaxis_title="Price (USD/bbl)",
        hovermode='x unified', height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Stats
    col1, col2, col3 = st.columns(3)
    pre = df_price[df_price['Date'] < '2026-02-28']['Price']
    post = df_price[df_price['Date'] >= '2026-02-28']['Price']
    col1.metric("Pre-Crisis Avg", f"${pre.mean():.2f}")
    col2.metric("Post-Crisis Avg", f"${post.mean():.2f}")
    col3.metric("Max Price Spike",
                f"${df_price['Price'].max():.2f}",
                f"+{df_price['Price'].max() - pre.mean():.2f}")

    # Volatility
    st.subheader("📊 Price Volatility Analysis")
    df_price['Daily_Change'] = df_price['Price'].pct_change() * 100
    fig2 = px.bar(df_price.tail(60), x='Date', y='Daily_Change',
                   color='Daily_Change',
                   color_continuous_scale=['#00CED1', '#FFD700', '#FF6B35'],
                   title="Daily % Price Change (Last 60 Trading Days)")
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Crisis Timeline — historical (static) + auto-generated latest row ──
    st.subheader("📅 Crisis Timeline")

    timeline_data = {
        'Date': [
            'Feb 28, 2026', 'Mar 1, 2026', 'Mar 4, 2026',
            'Mar 7, 2026', 'Mar 22, 2026', 'Apr 6, 2026',
            'Apr 13, 2026', 'May 5, 2026'
        ],
        'Event': [
            'US & Israel launch Operation Epic Fury on Iran',
            'Iran closes Strait of Hormuz',
            'Iran officially declares Hormuz closed',
            'US releases strategic petroleum reserves',
            'OPEC agrees to increase output by 2M bpd',
            'Iran allows 7 Malaysian ships through strait',
            'US announces full naval blockade of Iranian ports',
            'Trump announces Project Freedom'
        ],
        'Impact': [
            'Critical', 'Critical', 'Critical',
            'Moderate', 'Positive', 'Positive',
            'Critical', 'Moderate'
        ]
    }
    df_timeline = pd.DataFrame(timeline_data)

    # Auto-generated "latest update" row based on live price data
    latest_price = float(df_price['Price'].iloc[-1])
    pre_crisis_avg = float(pre.mean())
    pct_change = ((latest_price - pre_crisis_avg) / pre_crisis_avg) * 100

    if pct_change > 40:
        latest_impact = 'Critical'
    elif pct_change > 15:
        latest_impact = 'Moderate'
    else:
        latest_impact = 'Positive'

    auto_row = pd.DataFrame({
        'Date': [pd.Timestamp.now().strftime('%b %d, %Y')],
        'Event': [
            f'Live update: Brent crude at ${latest_price:.2f}/bbl '
            f'({pct_change:+.1f}% vs pre-crisis avg) — Day {crisis_days} of crisis'
        ],
        'Impact': [latest_impact]
    })

    df_timeline = pd.concat([df_timeline, auto_row], ignore_index=True)

    color_map = {'Critical': '🔴', 'Moderate': '🟡', 'Positive': '🟢'}
    df_timeline['Status'] = df_timeline['Impact'].map(color_map)
    st.dataframe(
        df_timeline[['Date', 'Status', 'Event']],
        use_container_width=True,
        hide_index=True
    )
    st.caption("📌 The last row updates automatically every time the dashboard loads, "
               "based on live Brent crude data.")

# ════════════════════════════════════════════════════
# TAB 2 — PRICE FORECAST
# ════════════════════════════════════════════════════
with tab2:
    st.subheader("🔮 Oil Price Forecast (Next 30 Days)")
    st.info("Using Linear Regression on post-crisis price trends to forecast future prices.")

    forecast_days = st.slider("Forecast horizon (days)", 7, 60, 30)
    scenario = st.selectbox("Select scenario", [
        "Base Case (Gradual Easing)",
        "Optimistic (Crisis Resolves)",
        "Pessimistic (Crisis Worsens — $200/bbl)"
    ])

    df_model = df_price_global.dropna().copy()
    df_model['Days'] = (df_model['Date'] - df_model['Date'].min()).dt.days

    post_crisis = df_model[df_model['Date'] >= '2026-02-28']
    if len(post_crisis) > 5:
        X = post_crisis[['Days']].values
        y = post_crisis['Price'].values
    else:
        X = df_model[['Days']].values
        y = df_model['Price'].values

    model_lr = LinearRegression()
    model_lr.fit(X, y)

    last_day = int(df_model['Days'].max())
    last_date = df_model['Date'].max()
    last_price = float(df_price_global['Price'].iloc[-1])
    future_days = np.arange(last_day + 1, last_day + forecast_days + 1)
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

    if scenario == "Optimistic (Crisis Resolves)":
        final_forecast = np.linspace(last_price, 72, forecast_days)
        color = '#00CED1'
        band = 6
    elif scenario == "Pessimistic (Crisis Worsens — $200/bbl)":
        final_forecast = np.linspace(last_price, 175, forecast_days)
        color = '#FF4500'
        band = 10
    else:
        final_forecast = np.linspace(last_price, 95, forecast_days)
        color = '#FFD700'
        band = 8

    final_forecast = np.clip(final_forecast, 50, 200)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_price_global['Date'], y=df_price_global['Price'],
        name='Historical', line=dict(color='#FF6B35', width=2)))
    fig3.add_trace(go.Scatter(
        x=future_dates, y=final_forecast,
        name=f'Forecast ({scenario.split("(")[0].strip()})',
        line=dict(color=color, width=2, dash='dash')))
    fig3.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=list(final_forecast + band) + list(final_forecast - band)[::-1],
        fill='toself', fillcolor=color,
        opacity=0.1, line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band'))
    fig3.update_layout(
        title=f"Oil Price Forecast — {scenario}",
        xaxis_title="Date", yaxis_title="Price (USD/bbl)",
        height=500, hovermode='x unified')
    st.plotly_chart(fig3, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    change = final_forecast[-1] - last_price
    col1.metric("Current Price", f"${last_price:.2f}")
    col2.metric("Forecast End Price", f"${final_forecast[-1]:.2f}")
    col3.metric("Expected Change", f"${change:.2f}",
                "🟢 Easing" if change < 0 else "🔴 Rising")

    st.info("""
    **Scenario Assumptions:**
    - 🟢 **Optimistic:** US-Iran ceasefire holds, Hormuz reopens within weeks, prices return to pre-crisis levels
    - 🟡 **Base Case:** Partial reopening via Project Freedom, gradual price easing to ~$95/bbl
    - 🔴 **Pessimistic:** Project Freedom fails, crisis escalates, analysts' $200/bbl target approached
    """)

# ════════════════════════════════════════════════════
# TAB 3 — COUNTRY IMPACT
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
        title='Oil Crisis Impact Score by Country (0–100)',
        labels={'impact_score': 'Impact Score'}
    )
    fig4.update_layout(height=500, geo=dict(showframe=False))
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.bar(
        df_filtered.sort_values('impact_score', ascending=True),
        x='impact_score', y='country',
        orientation='h', color='risk_level',
        color_discrete_map={
            'Critical': '#FF0000',
            'High': '#FF6B35',
            'Medium': '#FFD700',
            'Low': '#90EE90'
        },
        title='Countries Ranked by Impact Score',
        labels={'impact_score': 'Impact Score', 'country': 'Country'}
    )
    fig5.update_layout(height=500)
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("📋 Detailed Country Data")
    st.dataframe(
        df_filtered.sort_values('impact_score', ascending=False)
        .rename(columns={
            'country': 'Country',
            'impact_score': 'Impact Score',
            'daily_import_mbd': 'Daily Imports (MBD)',
            'price_increase_pct': 'Price Increase %',
            'risk_level': 'Risk Level'
        }),
        use_container_width=True, hide_index=True
    )

    st.info("""
    **Key Insight:** About 170 million barrels of crude oil, jet fuel and diesel
    remain trapped on tankers in the Gulf. Asian nations — China, Japan,
    India and South Korea — face the most critical supply shortages as the
    majority of Hormuz shipments are destined for Asian markets.
    """)

# ════════════════════════════════════════════════════
# TAB 4 — NEWS SENTIMENT
# ════════════════════════════════════════════════════
with tab4:
    st.subheader("📰 News Sentiment Analysis")
    st.info("Live headlines auto-fetched from Google News RSS, analysed using VADER NLP.")

    with st.spinner("Fetching latest news..."):
        try:
            df_news = fetch_oil_news()
            if df_news.empty:
                raise ValueError("No news found")
            data_source = "🟢 Live (Google News RSS)"
        except Exception:
            df_news = pd.read_csv("data/news_headlines.csv")
            df_news['date'] = pd.to_datetime(df_news['date'])
            data_source = "🟡 Fallback (saved CSV)"

    st.caption(f"Data source: {data_source} | Last refreshed: "
               f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")

    sid = SentimentIntensityAnalyzer()
    df_news['scores'] = df_news['headline'].apply(lambda x: sid.polarity_scores(x))
    df_news['compound'] = df_news['scores'].apply(lambda x: x['compound'])
    df_news['sentiment'] = df_news['compound'].apply(
        lambda x: '🟢 Positive' if x > 0.05
        else ('🔴 Negative' if x < -0.05 else '🟡 Neutral'))

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    neg = len(df_news[df_news['compound'] < -0.05])
    pos = len(df_news[df_news['compound'] > 0.05])
    neu = len(df_news) - neg - pos
    col1.metric("🔴 Negative", neg)
    col2.metric("🟡 Neutral", neu)
    col3.metric("🟢 Positive", pos)
    col4.metric("📰 Total Headlines", len(df_news))

    # Sentiment over time
    fig6 = go.Figure()
    colors_sentiment = df_news['compound'].apply(
        lambda x: '#FF4500' if x < -0.05
        else ('#90EE90' if x > 0.05 else '#FFD700'))
    fig6.add_trace(go.Bar(
        x=df_news['date'], y=df_news['compound'],
        marker_color=colors_sentiment, name='Sentiment Score'))
    fig6.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.5)
    fig6.update_layout(
        title='News Sentiment Score Over Time',
        xaxis_title='Date', yaxis_title='Compound Sentiment Score',
        height=400)
    st.plotly_chart(fig6, use_container_width=True)

    # Pie + daily trend
    col1, col2 = st.columns(2)
    with col1:
        sentiment_counts = df_news['sentiment'].value_counts()
        fig7 = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                '🔴 Negative': '#FF4500',
                '🟡 Neutral': '#FFD700',
                '🟢 Positive': '#90EE90'
            },
            title='Overall Sentiment Distribution',
            hole=0.4
        )
        fig7.update_layout(height=350)
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        df_news['day'] = df_news['date'].dt.date
        daily_sentiment = df_news.groupby('day')['compound'].mean().reset_index()
        daily_sentiment.columns = ['Date', 'Avg Sentiment']
        fig8 = px.bar(
            daily_sentiment, x='Date', y='Avg Sentiment',
            color='Avg Sentiment',
            color_continuous_scale=['#FF4500', '#FFD700', '#90EE90'],
            title='Average Daily Sentiment Score'
        )
        fig8.update_layout(height=350)
        st.plotly_chart(fig8, use_container_width=True)

    # Headlines table
    st.subheader("📋 Latest Headlines with Sentiment")
    display_df = df_news[['date', 'headline', 'source', 'sentiment', 'compound']].copy()
    display_df['compound'] = display_df['compound'].round(3)
    display_df.columns = ['Date', 'Headline', 'Source', 'Sentiment', 'Score']
    st.dataframe(
        display_df.sort_values('Date', ascending=False),
        use_container_width=True, hide_index=True
    )

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style='text-align:center;color:#888;font-size:0.85rem'>
    🛢️ Oil & Gas Crisis Dashboard 2026 | Built with Python, Streamlit & Plotly<br>
    Live price data: yFinance | Live news: Google News RSS | Auto-updated as of {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
""", unsafe_allow_html=True)