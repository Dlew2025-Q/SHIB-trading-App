# backend/signal_logger.py

import asyncio
import os
import httpx
import asyncpg
import json
import numpy as np
from datetime import datetime

# --- Configuration ---
# Load environment variables from the Render environment
DATABASE_URL = os.getenv("DATABASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

db_pool = None

# --- Database Functions ---
async def connect_to_db():
    """Establishes a connection pool to the PostgreSQL database."""
    global db_pool
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL environment variable is not set.")
        return
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL, timeout=10)
        print("Signal logger connected to PostgreSQL database.")
    except Exception as e:
        print(f"CRITICAL ERROR: Logger could not connect to database: {e}")
        db_pool = None

async def save_signal_to_db(trade_data):
    """Saves a generated trade signal to the database."""
    if not db_pool:
        print("Cannot save signal, no database connection.")
        return
    try:
        # A new trade is always 'pending' until its outcome is checked later
        status = 'pending'
        await db_pool.execute('''
            INSERT INTO simulated_trades (id, signal_type, entry_price, take_profit_price, stop_loss_price, position_size, status, timestamp, ai_reasoning)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ''',
        trade_data['id'], trade_data['signalType'], float(trade_data['entryPrice']), float(trade_data['takeProfitPrice']),
        float(trade_data['stopLossPrice']), float(trade_data['positionSize']), status,
        trade_data['timestamp'], trade_data['aiReasoning']
        )
        print(f"Successfully logged {trade_data['signalType']} signal to database.")
    except Exception as e:
        print(f"Error saving signal to DB: {e}")

# --- AI and Data Fetching Functions ---
async def get_market_data():
    """Fetches the latest price and historical OHLC data from CoinGecko."""
    try:
        async with httpx.AsyncClient() as client:
            headers = {"x-cg-demo-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
            
            # Fetch current price
            price_url = 'https://api.coingecko.com/api/v3/coins/shiba-inu?market_data=true'
            price_res = await client.get(price_url, headers=headers, timeout=10)
            price_res.raise_for_status()
            current_price = price_res.json()['market_data']['current_price']['usd']

            # Fetch historical data (30 days needed for SMAs)
            ohlc_url = f"https://api.coingecko.com/api/v3/coins/shiba-inu/ohlc?vs_currency=usd&days=30"
            ohlc_res = await client.get(ohlc_url, headers=headers, timeout=10)
            ohlc_res.raise_for_status()
            
            return {"current_price": current_price, "historical_ohlc": ohlc_res.json()}
    except Exception as e:
        print(f"Error fetching data from CoinGecko: {e}")
        return None

async def get_ai_trade_signal(market_data):
    """Calls the Gemini AI to get a trade signal based on the latest tuned strategy."""
    def calculate_sma(series, period):
        if len(series) < period: return None
        return sum(series[-period:]) / period

    def calculate_atr(ohlc_data, period=14):
        if len(ohlc_data) < period + 1: return None
        true_ranges = [max(d[2] - d[3], abs(d[2] - ohlc_data[i-1][4]), abs(d[3] - ohlc_data[i-1][4])) for i, d in enumerate(ohlc_data) if i > 0]
        if not true_ranges: return None
        return np.mean(true_ranges[-period:])

    closing_prices = [d[4] for d in market_data['historical_ohlc']]
    price_sma_10 = calculate_sma(closing_prices, 10)
    atr_14 = calculate_atr(market_data['historical_ohlc'], 14)

    if not all([price_sma_10, atr_14]):
        print("Could not calculate necessary technical indicators.")
        return None

    yesterday_ohlc = market_data['historical_ohlc'][-1]
    yesterday_open, yesterday_close = yesterday_ohlc[1], yesterday_ohlc[4]

    prompt_parts = [
        "You are a trading analyst. Your task is to evaluate a momentum strategy with newly tuned parameters and generate a signal if the conditions are met.",
        "\n--- Strategy Rules (Tuned Version) ---",
        "1. **Regime Filter:** LONG if Current Price > 10-Day SMA; SHORT if Current Price < 10-Day SMA.",
        "2. **Entry Signal:** Previous Green Candle (Close > Open) for LONG; Previous Red Candle (Close < Open) for SHORT.",
        "3. **Dynamic Exits (ATR - Tuned):** Calculate exit points using the provided 14-day Average True Range (ATR).",
        "   - **Take-Profit:** Entry Price +/- (2.25 * ATR)",
        "   - **Stop-Loss:** Entry Price -/+ (1.5 * ATR)",
        "4. **Final Decision:** A trade signal is only generated if ALL conditions for that direction are met. If any condition fails, you MUST return a 'NEUTRAL' signal.",

        "\n--- Data Provided for Analysis ---",
        f"- Current Price (for Entry): ${market_data['current_price']}",
        f"- Previous Day's Open: ${yesterday_open}",
        f"- Previous Day's Close: ${yesterday_close}",
        f"- 10-Day Price SMA: ${price_sma_10:.8f}",
        f"- 14-Day ATR: ${atr_14:.8f}",
        
        "\n--- Your Task ---",
        "Follow the tuned rules to generate a signal and its corresponding exit prices. Provide your response in the following strict JSON format.",
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
        return json.loads(result['candidates'][0]['content']['parts'][0]['text'])
    except Exception as e:
        print(f"Error getting AI signal: {e}")
        return None

# --- Main Logging Loop ---
async def run_logging_loop():
    """The main loop that runs indefinitely, fetching and logging signals."""
    while True:
        print(f"\n--- New Logging Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        
        # 1. Fetch market data
        market_data = await get_market_data()
        if not market_data:
            print("Failed to get market data. Retrying in 30 minutes.")
            await asyncio.sleep(1800)
            continue
        
        # 2. Get AI signal
        ai_signal = await get_ai_trade_signal(market_data)
        if not ai_signal or ai_signal.get('signal_type') == 'NEUTRAL':
            print("AI signal is NEUTRAL. No signal logged.")
            await asyncio.sleep(1800) # Sleep for 30 minutes
            continue

        # 3. Log the signal to the database
        print(f"AI generated a {ai_signal['signal_type']} signal. Logging to database.")
        trade_to_log = {
            "id": int(datetime.now().timestamp() * 1000),
            "signalType": ai_signal['signal_type'],
            "entryPrice": market_data['current_price'],
            "takeProfitPrice": ai_signal['take_profit_price'],
            "stopLossPrice": ai_signal['stop_loss_price'],
            "positionSize": 0, # Position size is 0 as we are not executing
            "timestamp": int(datetime.now().timestamp() * 1000),
            "aiReasoning": ai_signal['reasoning'],
        }
        await save_signal_to_db(trade_to_log)
        
        # 4. Wait for the next cycle
        print("Cycle complete. Sleeping for 30 minutes.")
        await asyncio.sleep(1800)

async def main():
    """Initializes the database connection and starts the logging loop."""
    await connect_to_db()
    if not db_pool:
        print("Exiting logger due to database connection failure.")
        return
    await run_logging_loop()

if __name__ == "__main__":
    print("Starting SHIB Signal Logger...")
    if not all([DATABASE_URL, GEMINI_API_KEY]):
        print("CRITICAL ERROR: DATABASE_URL or GEMINI_API_KEY environment variables are missing. Exiting.")
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("Logger stopped by user.")
