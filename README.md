## Crypto Marget Dashboard

A Streamlit dashboard for real-time cryptocurrency market analysis using the CoinGecko API.

This project visualizes market trends, highlights top gainers/losers, and includes a **simple ML model** for next-day price prediction.


#### Features : 
- **Live market data** for top cryptocurrencies
- **Interactive leaderboard** for gainers/losers tables
- **Search and filter** coins
- **Historical trend charts** using Plotly
- **Machine Learning** (Linear Regression) for basic price prediction
- **CSV download** for market data


#### Tech stack : 
- **Python**
- **Streamlit** – web app framework
- **Plotly** – interactive visualizations
- **scikit-learn** – ML for price prediction
- **Pandas** – data manipulation

#### Data Source
All data is fetched from the **CoinGecko free API**.  
Data is cached for **3 minutes** to avoid rate limits and improve performance.

#### Installation & Usage

1. **Clone the repo**  
git clone https://github.com/ayush-skech/crypto-dashboard.git
cd crypto-dashboard