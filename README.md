# Oil & Gas Crisis Dashboard 2026 🛢️

an interactive data science dashboard i built to analyze the ongoing 2026 oil crisis 
triggered by the Strait of Hormuz disruption. pulls live oil price data and combines 
it with ML forecasting, country impact analysis and news sentiment — all in one place.

---

## what it does

the dashboard has 4 tabs:

- **price analysis** — live brent crude price chart with crisis event markers, 
moving averages and daily volatility since sep 2025
- **price forecast** — 3 scenario forecasting (optimistic, base case, pessimistic) 
using linear regression with confidence bands
- **country impact map** — world choropleth showing which countries are most affected 
by the crisis with risk levels and import data
- **news sentiment** — VADER NLP analysis on 31 real crisis headlines showing how 
media sentiment shifted over time

---

## some context on the crisis

on feb 28 2026 the US and Israel launched joint air strikes on Iran which caused brent 
crude to spike from around $65 to over $112 in a matter of weeks. the Strait of Hormuz 
which handles roughly 20% of the world's oil supply was effectively shut down.

this dashboard tracks exactly that story through data

---

## tech stack

- **python** — main language
- **streamlit** — web dashboard framework
- **plotly** — interactive charts and world map
- **yfinance** — live brent crude price data (completely free)
- **scikit-learn** — linear regression for price forecasting
- **VADER (nltk)** — sentiment analysis on news headlines
- **pandas / numpy** — data processing

---



## project structure
```
oil-gas-dashboard/
├── app.py                    # main streamlit dashboard
├── data/
│   ├── country_impact.csv    # impact scores for 20 countries
│   └── news_headlines.csv    # 31 real crisis news headlines
├── requirements.txt

```

---

## key findings from the data

- brent crude jumped **+48%** from pre-crisis average ($65) to post-crisis ($96)
- **china, japan and india** are the most critically impacted countries
- news sentiment was overwhelmingly negative in early march but started recovering 
by march 22 as OPEC announced output increases
- base case forecast shows prices gradually easing to around $88 by april 2026

---

## what i want to add later

- more countries in the impact map (africa, southeast asia)
- correlation analysis between news sentiment and price movements
- supply vs demand gap chart
- automatic news scraping instead of manual headlines

---

## what i learnt

honestly the most interesting part was seeing how closely the sentiment chart matched 
the actual price movements. when headlines turned positive around march 7 the price 
dipped too. did not expect that pattern to show up so clearly in the data.

also working with live financial data using yfinance was new for me and way simpler 
than i expected

---

made with 💖 by claude and Ayush
