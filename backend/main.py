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
            "high_24h": market_data.get('high_24h', {}).get('usd'),
            "low_24h": market_data.get('low_24h', {}).get('usd'),
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
    news_items = [
        {"title": "Latest Shiba Inu News", "url": "https://news.google.com/search?q=Shiba%20Inu%20coin", "source": "Google News"},
        {"title": "Shiba Inu News on CoinDesk", "url": "https://www.coindesk.com/search?q=shiba%20inu", "source": "CoinDesk"},
        {"title": "Shiba Inu on CoinTelegraph", "url": "https://cointelegraph.com/search?query=shiba%20inu", "source": "CoinTelegraph"},
        {"title": "SHIB News on Decrypt", "url": "https://decrypt.co/search/shiba%20inu", "source": "Decrypt"},
        {"title": "Today's Crypto News", "url": "https://news.google.com/search?q=cryptocurrency", "source": "Google News"}
    ]
    return {"news": news_items[:limit]}

# --- AI SIGNAL WITH UPGRADED STRATEGY ---
@app.post("/ai-trade-signal")
async def ai_trade_signal(request_body: dict):
    current_price = request_body.get("current_price")
    historical_ohlc = request_body.get("historical_ohlc", [])
    
    if not all([current_price, historical_ohlc]):
        raise HTTPException(status_code=400, detail="Missing required data for AI signal.")

    def calculate_atr(ohlc_data, period=14):
        if len(ohlc_data) < period + 1: return None
        true_ranges = [max(d[2] - d[3], abs(d[2] - ohlc_data[i-1][4]), abs(d[3] - ohlc_data[i-1][4])) for i, d in enumerate(ohlc_data) if i > 0]
        if not true_ranges: return None
        return np.mean(true_ranges[-period:])

    # Calculate indicators based on the last 30 days of data
    if len(historical_ohlc) < 4: # Need at least 3 previous candles + current
        raise HTTPException(status_code=500, detail="Not enough historical data for analysis.")

    atr_14 = calculate_atr(historical_ohlc, 14)
    if not atr_14:
        raise HTTPException(status_code=500, detail="Could not calculate ATR.")

    # Get the high and low of the previous 3 candles
    high_3_day = max(d[2] for d in historical_ohlc[-4:-1])
    low_3_day = min(d[3] for d in historical_ohlc[-4:-1])

    prompt_parts = [
        "You are a trading analyst. Your task is to evaluate a breakout strategy and generate a signal if the conditions are met.",
        "\n--- Strategy Rules (Version 3.0) ---",
        "1. **Entry Signal (Breakout):**",
        "   - **LONG:** Enter ONLY if the Current Price is GREATER than the High of the previous 3 candles.",
        "   - **SHORT:** Enter ONLY if the Current Price is LESS than the Low of the previous 3 candles.",
        "2. **Dynamic Exits (ATR):** Calculate exit points using the provided 14-day Average True Range (ATR).",
        "   - **Take-Profit:** Entry Price +/- (1.5 * ATR)",
        "   - **Stop-Loss:** Entry Price -/+ (1.0 * ATR)",
        "3. **Final Decision:** A trade signal is only generated if a breakout condition is met. If neither condition is met, you MUST return a 'NEUTRAL' signal.",

        "\n--- Data Provided for Analysis ---",
        f"- Current Price (for Entry): ${current_price}",
        f"- High of previous 3 candles: ${high_3_day:.8f}",
        f"- Low of previous 3 candles: ${low_3_day:.8f}",
        f"- 14-Day ATR: ${atr_14:.8f}",
        
        "\n--- Your Task ---",
        "Follow the rules to generate a signal and its corresponding exit prices. Provide your response in the following strict JSON format.",
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
                }, "required": ["signal_type", "reasoning"]
            }
        }
    }
    
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(gemini_api_url, json=payload, timeout=45.0)
            response.raise_for_status()
            result = response.json()
        ai_response_text = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(ai_response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during AI signal generation: {e}")

@app.post("/ai-strategy-review")
async def ai_strategy_review(trades: list[dict]):
    if not trades:
        raise HTTPException(status_code=400, detail="No trade history provided for review.")

    completed_trades = [t for t in trades if t.get('status') in ['win', 'loss']]
    if not completed_trades:
        return {"backtest_result": "Not enough completed trades to run a simulation.", "observations": "N/A", "recommendations": "N/A", "suggested_position_size": "N/A", "suggested_risk_reward_ratio": "N/A"}

    completed_trades.reverse()
    trade_history_str = "\n".join([f"- Status: {t['status']}" for t in completed_trades])

    prompt_parts = [
        "You are an expert quantitative trading strategist. Your task is to analyze the performance of a trading algorithm and provide a full analysis including a portfolio simulation.",
        "\n--- Current Strategy Rules ---",
        "1. **Entry Signal:** LONG if Price > High of last 3 candles; SHORT if Price < Low of last 3 candles.",
        "2. **Exits:** Take-Profit at 1.5 * ATR; Stop-Loss at 1.0 * ATR. (This is a 1.5-to-1 Reward/Risk Ratio).",
        "\n--- Recent Completed Trade History (Oldest to Newest) ---",
        trade_history_str,
        "\n--- Your Analysis Task (MUST COMPLETE ALL PARTS) ---",
        "1. **Portfolio Simulation:** Run a simulation with a starting bankroll of $10,000. For each trade, invest 2% of the current bankroll. A 'win' earns a profit based on the 1.5 reward/risk ratio (e.g., +3%); a 'loss' incurs a loss based on the risk (e.g., -2%). Calculate the final bankroll and total percentage return.",
        "2. **Identify Patterns:** Analyze the losing trades. Is there a common reason for failure? (e.g., false breakouts, stop-loss too tight).",
        "3. **Suggest Rule Adjustments:** Based on the patterns, suggest specific, numerical adjustments to the strategy rules.",
        "4. **Suggest Position Sizing & Risk/Reward:** Based on the simulation, recommend a position size (% of bankroll) and a Reward/Risk ratio.",
        "5. **Format Response:** You MUST provide your complete analysis in the following strict JSON format, filling all five keys.",
    ]

    prompt = "\n".join(prompt_parts)

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "backtest_result": {"type": "STRING", "description": "The final summary sentence of the backtest result."},
                    "observations": {"type": "STRING", "description": "A summary of the key patterns observed in the losing trades."},
                    "recommendations": {"type": "STRING", "description": "Specific, actionable suggestions for rule adjustments."},
                    "suggested_position_size": {"type": "STRING", "description": "A suggested bankroll percentage to risk per trade, e.g., '1-2%'"},
                    "suggested_risk_reward_ratio": {"type": "STRING", "description": "A suggested Reward-to-Risk ratio, e.g., '1.5:1'"}
                }, "required": ["backtest_result", "observations", "recommendations", "suggested_position_size", "suggested_risk_reward_ratio"]
            }
        }
    }

    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(gemini_api_url, json=payload, timeout=60.0)
            response.raise_for_status()
            result = response.json()
        ai_response_text = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(ai_response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during AI strategy review: {e}")

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
    entry_price = trade_details.get("entry_price")
    take_profit_price = trade_details.get("take_profit_price")
    stop_loss_price = trade_details.get("stop_loss_price")
    signal_type = trade_details.get("signal_type")
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

@app.post("/purge-db")
async def purge_db():
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection not established.")
    
    try:
        async with db_pool.acquire() as conn:
            await conn.execute('TRUNCATE TABLE simulated_trades;')
        print("Database table 'simulated_trades' has been purged.")
        return {"message": "Trade history has been successfully cleared."}
    except Exception as e:
        print(f"CRITICAL ERROR: Could not purge database table: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear trade history.")
