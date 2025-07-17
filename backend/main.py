# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import asyncio
import asyncpg
import json
from datetime import datetime, timedelta
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://shib-trading-app-front-end.onrender.com"], # Use your specific frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
if not COINGECKO_API_KEY:
    print("WARNING: COINGECKO_API_KEY environment variable is not set!")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable is not set!")
else:
    print(f"INFO: GEMINI_API_KEY loaded (starts with: {GEMINI_API_KEY[:5]}...)")

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
if not RAPIDAPI_KEY:
    print("WARNING: RAPIDAPI_KEY environment variable is not set! Crypto news will not work.")


db_pool = None

# --- Database Setup Functions ---
async def connect_to_db():
    global db_pool
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL environment variable is NOT SET! Database connection will fail.")
        db_pool = None
        return

    try:
        print(f"Attempting to connect to database using URL: {DATABASE_URL[:30]}...")
        db_pool = await asyncpg.create_pool(DATABASE_URL, timeout=10)
        print("Successfully connected to PostgreSQL database pool.")
        await create_trades_table()
    except Exception as e:
        print(f"CRITICAL ERROR: Could not connect to or initialize database: {e}")
        db_pool = None
        raise

async def disconnect_from_db():
    global db_pool
    if db_pool:
        await db_pool.close()
        print("Disconnected from PostgreSQL database.")

async def create_trades_table():
    if not db_pool:
        print("WARNING: Skipping create_trades_table as db_pool is not established.")
        return

    async with db_pool.acquire() as conn:
        try:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS simulated_trades (
                    id BIGINT PRIMARY KEY,
                    signal_type VARCHAR(10) NOT NULL,
                    entry_price NUMERIC(20, 8) NOT NULL,
                    take_profit_price NUMERIC(20, 8) NOT NULL,
                    stop_loss_price NUMERIC(20, 8) NOT NULL,
                    position_size NUMERIC(20, 8) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    outcome_price NUMERIC(20, 8),
                    profit_loss NUMERIC(20, 8),
                    timestamp BIGINT NOT NULL,
                    ai_reasoning TEXT,
                    sentiment_score NUMERIC(10, 8)
                );
            ''')
            print("simulated_trades table checked/created with full schema.")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not create/update simulated_trades table schema: {e}")
            raise

# --- FastAPI Lifecycle Events (connect/disconnect DB) ---
@app.on_event("startup")
async def startup_event():
    print("Application startup event triggered.")
    try:
        await connect_to_db()
        print("Startup: Database connection attempt completed.")
    except Exception as e:
        print(f"FATAL ERROR during application startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutdown event triggered.")
    await disconnect_from_db()

# --- Helper function to make CoinGecko API calls ---
async def fetch_coingecko_data(url: str):
    await asyncio.sleep(0.5)
    headers = {}
    if COINGECKO_API_KEY:
        headers["x-cg-demo-api-key"] = COINGECKO_API_KEY
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"CoinGecko API error: {e.response.text}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Network error fetching CoinGecko data: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- Helper function to fetch crypto news ---
async def fetch_crypto_news(limit: int = 5):
    if not RAPIDAPI_KEY:
        print("ERROR: RAPIDAPI_KEY is not set. Cannot fetch crypto news.")
        return []
    await asyncio.sleep(0.5)
    news_url = f"https://coin-echo-crypto-news-aggregator-and-sentiment-analysis.p.rapidapi.com/api/news?limit={limit}"
    headers = {
        'x-rapidapi-host': 'coin-echo-crypto-news-aggregator-and-sentiment-analysis.p.rapidapi.com',
        'x-rapidapi-key': RAPIDAPI_KEY
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(news_url, headers=headers, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            if data and isinstance(data, list):
                return [{"title": item.get("title"), "url": item.get("article_url"), "source": item.get("source")} for item in data]
            return []
        except Exception as e:
            print(f"Error fetching Crypto News: {e}")
            return []

# --- Backend Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the SHIB Trading Analysis Backend API!"}

@app.get("/shib-prices")
async def get_shib_prices():
    market_data_url = 'https://api.coingecko.com/api/v3/coins/shiba-inu?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false'
    try:
        data = await fetch_coingecko_data(market_data_url)
        market_data = data.get('market_data', {})
        return {
            "current_price": market_data.get('current_price', {}).get('usd'),
            "price_change_24h": market_data.get('price_change_percentage_24h'),
            "market_cap": market_data.get('market_cap', {}).get('usd'),
            "total_volume": market_data.get('total_volume', {}).get('usd'),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process SHIB prices: {e}")

@app.get("/shib-historical-data")
async def get_shib_historical_data():
    ohlc_url = f"https://api.coingecko.com/api/v3/coins/shiba-inu/ohlc?vs_currency=usd&days=30"
    volume_url = f"https://api.coingecko.com/api/v3/coins/shiba-inu/market_chart?vs_currency=usd&days=30"
    try:
        ohlc_data = await fetch_coingecko_data(ohlc_url)
        volume_data = await fetch_coingecko_data(volume_url)
        return {
            "ohlc": ohlc_data,
            "volumes": volume_data.get('total_volumes', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process historical data: {e}")

@app.get("/crypto-news/{limit}")
async def get_crypto_news_endpoint(limit: int):
    news_items = await fetch_crypto_news(limit)
    return {"news": news_items}

# --- NEW MULTI-FACTOR AI SIGNAL ---
@app.post("/ai-trade-signal")
async def ai_trade_signal(request_body: dict):
    current_price = request_body.get("current_price")
    historical_ohlc = request_body.get("historical_ohlc", [])
    historical_volumes = request_body.get("historical_volumes", [])

    if not all([current_price, historical_ohlc, historical_volumes]):
        raise HTTPException(status_code=400, detail="Missing required data for AI signal.")

    # --- Technical Analysis Helper Functions ---
    def calculate_sma(series, period):
        if len(series) < period: return None
        return sum(series[-period:]) / period

    def calculate_atr(ohlc_data, period=14):
        if len(ohlc_data) < period + 1: return None
        true_ranges = []
        for i in range(1, len(ohlc_data)):
            high = ohlc_data[i][2]
            low = ohlc_data[i][3]
            prev_close = ohlc_data[i-1][4]
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        if not true_ranges: return None
        return np.mean(true_ranges[-period:])

    # --- Calculate Indicators ---
    closing_prices = [d[4] for d in historical_ohlc]
    volumes = [v[1] for v in historical_volumes]
    
    price_sma_10 = calculate_sma(closing_prices, 10)
    volume_sma_10 = calculate_sma(volumes, 10)
    atr_14 = calculate_atr(historical_ohlc, 14)

    if not all([price_sma_10, volume_sma_10, atr_14]):
        raise HTTPException(status_code=500, detail="Could not calculate necessary technical indicators.")

    yesterday_ohlc = historical_ohlc[-1]
    yesterday_open = yesterday_ohlc[1]
    yesterday_close = yesterday_ohlc[4]
    yesterday_volume = volumes[-1]

    # --- Construct the new super-prompt ---
    prompt_parts = [
        "You are a professional algorithmic trading analyst. Your task is to generate a trade signal for SHIB based on a specific multi-factor strategy. You must follow the rules precisely.",
        "\n--- Strategy Rules ---",
        "1. **Market Regime Filter:** Use the 10-day Simple Moving Average (SMA) of the price. A LONG trade is ONLY allowed if the current price is ABOVE the 10-day SMA. A SHORT trade is ONLY allowed if the current price is BELOW the 10-day SMA.",
        "2. **Volume Confirmation:** A signal is only valid if the previous day's trading volume was GREATER than the 10-day SMA of volume. This confirms market conviction.",
        "3. **Entry Signal:** The basic signal comes from the previous day's candle color. Green candle (Close > Open) for a potential LONG. Red candle (Close < Open) for a potential SHORT.",
        "4. **Dynamic Exits (ATR):** All exit points must be calculated using the provided 14-day Average True Range (ATR).",
        "   - **Take-Profit:** Entry Price +/- (1.5 * ATR)",
        "   - **Stop-Loss:** Entry Price -/+ (1.0 * ATR)",
        "5. **Final Decision:** A trade signal is only generated if ALL conditions (Regime, Volume, Entry) are met. If any condition fails, you MUST return a 'NEUTRAL' signal.",

        "\n--- Data Provided for Analysis ---",
        f"- Current Price (for Entry): ${current_price}",
        f"- Previous Day's Open: ${yesterday_open}",
        f"- Previous Day's Close: ${yesterday_close}",
        f"- Previous Day's Volume: {yesterday_volume:,.0f}",
        f"- 10-Day Price SMA: ${price_sma_10:.8f}",
        f"- 10-Day Volume SMA: {volume_sma_10:,.0f}",
        f"- 14-Day ATR: ${atr_14:.8f}",
        
        "\n--- Your Task ---",
        "1. Check if a LONG signal is valid: Is Previous Day Green? Is Volume Confirmed? Is Current Price > 10-Day Price SMA?",
        "2. Check if a SHORT signal is valid: Is Previous Day Red? Is Volume Confirmed? Is Current Price < 10-Day Price SMA?",
        "3. If a valid signal exists, calculate the take_profit_price and stop_loss_price using the ATR.",
        "4. If no signal is valid, signal NEUTRAL.",
        "5. Provide your response in the following strict JSON format. Provide a concise reasoning based on the rules.",
    ]
    
    prompt = "\n".join(prompt_parts)
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "signal_type": {"type": "STRING", "enum": ["LONG", "SHORT", "NEUTRAL"]},
                    "reasoning": {"type": "STRING"},
                    "take_profit_price": {"type": "NUMBER"},
                    "stop_loss_price": {"type": "NUMBER"}
                },
                "required": ["signal_type", "reasoning"]
            }
        }
    }
    
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(gemini_api_url, json=payload, timeout=45.0)
            response.raise_for_status()
            result = response.json()
        
        ai_response_text = result['candidates'][0]['content']['parts'][0]['text']
        ai_response = json.loads(ai_response_text)
        
        return ai_response

    except Exception as e:
        print(f"Error during AI signal generation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during AI signal generation: {e}")

# --- Other Endpoints (save-trade, get-all-trades, check-signal-outcome) remain the same ---

@app.post("/save-trade")
async def save_trade(trade_data: dict):
    if not db_pool: raise HTTPException(status_code=500, detail="Database connection not established.")
    try:
        await db_pool.execute('''
            INSERT INTO simulated_trades (id, signal_type, entry_price, take_profit_price, stop_loss_price, position_size, status, outcome_price, profit_loss, timestamp, ai_reasoning, sentiment_score)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (id) DO UPDATE SET
                signal_type = EXCLUDED.signal_type, entry_price = EXCLUDED.entry_price, take_profit_price = EXCLUDED.take_profit_price,
                stop_loss_price = EXCLUDED.stop_loss_price, position_size = EXCLUDED.position_size, status = EXCLUDED.status,
                outcome_price = EXCLUDED.outcome_price, profit_loss = EXCLUDED.profit_loss, timestamp = EXCLUDED.timestamp,
                ai_reasoning = EXCLUDED.ai_reasoning, sentiment_score = EXCLUDED.sentiment_score;
        ''',
        trade_data['id'], trade_data['signalType'], float(trade_data['entryPrice']), float(trade_data['takeProfitPrice']),
        float(trade_data['stopLossPrice']), float(trade_data['positionSize']), trade_data['status'],
        trade_data.get('outcomePrice'), trade_data.get('profitLoss'), trade_data['timestamp'],
        trade_data['aiReasoning'], trade_data.get('sentimentScore')
        )
        return {"message": "Trade saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error saving trade: {e}")

@app.get("/get-all-trades")
async def get_all_trades():
    if not db_pool: raise HTTPException(status_code=500, detail="Database connection not established.")
    try:
        records = await db_pool.fetch('SELECT * FROM simulated_trades ORDER BY timestamp DESC;')
        return {"trades": [dict(record) for record in records]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error fetching trades: {e}")

@app.post("/check-signal-outcome")
async def check_signal_outcome(trade_details: dict):
    # This function remains the same as it correctly checks outcomes based on saved TP/SL levels.
    entry_price = trade_details.get("entryPrice")
    take_profit_price = trade_details.get("takeProfitPrice")
    stop_loss_price = trade_details.get("stopLossPrice")
    signal_type = trade_details.get("signalType")
    timestamp = trade_details.get("timestamp")
    trade_id = trade_details.get("id")

    if not all([entry_price, take_profit_price, stop_loss_price, signal_type, timestamp, trade_id]):
        raise HTTPException(status_code=400, detail="Missing trade details for outcome check.")

    try:
        from_ts = timestamp / 1000
        to_ts = from_ts + (2 * 24 * 60 * 60)
        chart_url = f"https://api.coingecko.com/api/v3/coins/shiba-inu/market_chart/range?vs_currency=usd&from={from_ts}&to={to_ts}"
        data = await fetch_coingecko_data(chart_url)
        prices = data.get('prices', [])

        if not prices or len(prices) < 2:
            return {"status": "pending", "outcomePrice": entry_price, "profitLoss": 0, "message": "Not enough data."}

        outcome, final_price, profit_loss = 'pending', entry_price, 0
        for _, price in prices:
            final_price = price
            if signal_type == 'LONG':
                if price >= take_profit_price: outcome = 'win'; break
                if price <= stop_loss_price: outcome = 'loss'; break
            else: # SHORT
                if price <= take_profit_price: outcome = 'win'; break
                if price >= stop_loss_price: outcome = 'loss'; break
        
        await db_pool.execute(
            'UPDATE simulated_trades SET status = $1, outcome_price = $2, profit_loss = $3 WHERE id = $4;',
            outcome, final_price, profit_loss, trade_id
        )
        return {"status": outcome, "outcomePrice": final_price, "profitLoss": profit_loss}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check signal outcome: {e}")
