# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import asyncio
import asyncpg # New import for async PostgreSQL driver
import json # To handle JSON serialization for database storage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
if not COINGECKO_API_KEY:
    print("WARNING: COINGECKO_API_KEY environment variable is not set!")

GEMINI_API_KEY = "" # Handled by Canvas or can be set as env var for direct deployment

# Database connection pool (best practice for async apps)
db_pool = None

# --- Database Setup Functions ---
async def connect_to_db():
    global db_pool
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL environment variable is not set!")
        return

    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL)
        print("Successfully connected to PostgreSQL database.")
        await create_trades_table() # Ensure table exists on startup
    except Exception as e:
        print(f"ERROR: Could not connect to database: {e}")

async def disconnect_from_db():
    global db_pool
    if db_pool:
        await db_pool.close()
        print("Disconnected from PostgreSQL database.")

async def create_trades_table():
    async with db_pool.acquire() as conn:
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
                timestamp BIGINT NOT NULL
            );
        ''')
        print("simulated_trades table checked/created.")

# --- FastAPI Lifecycle Events (connect/disconnect DB) ---
@app.on_event("startup")
async def startup_event():
    await connect_to_db()

@app.on_event("shutdown")
async def shutdown_event():
    await disconnect_from_db()

# --- Helper function to make CoinGecko API calls ---
async def fetch_coingecko_data(url: str):
    await asyncio.sleep(0.5) # Wait for 0.5 seconds

    headers = {}
    if COINGECKO_API_KEY:
        headers["x-cg-demo-api-key"] = COINGECKO_API_KEY

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error fetching CoinGecko data from {url}: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"CoinGecko API error: {e.response.text}")
        except httpx.RequestError as e:
            print(f"Network Error fetching CoinGecko data from {url}: {e}")
            raise HTTPException(status_code=500, detail=f"Network error fetching CoinGecko data: {e}")
        except Exception as e:
            print(f"Unexpected Error fetching CoinGecko data from {url}: {e}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- Backend Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to the SHIB Trading Analysis Backend API!"}

@app.get("/shib-prices")
async def get_shib_prices():
    print("Received request for /shib-prices")
    market_data_url = 'https://api.coingecko.com/api/v3/coins/shiba-inu?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false'
    
    try:
        data = await fetch_coingecko_data(market_data_url)
        market_data = data.get('market_data', {})
        print("Successfully fetched market data from CoinGecko.")

        return {
            "current_price": market_data.get('current_price', {}).get('usd'),
            "price_change_24h": market_data.get('price_change_percentage_24h'),
            "market_cap": market_data.get('market_cap', {}).get('usd'),
            "total_volume": market_data.get('total_volume', {}).get('usd'),
            "circulating_supply": market_data.get('circulating_supply'),
            "ath": market_data.get('ath', {}).get('usd')
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /shib-prices endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process SHIB prices: {e}")

@app.get("/shib-historical-data")
async def get_shib_historical_data():
    print("Received request for /shib-historical-data")
    chart_url = f"https://api.coingecko.com/api/v3/coins/shiba-inu/market_chart?vs_currency=usd&days=30"
    
    try:
        data = await fetch_coingecko_data(chart_url)
        print("Successfully fetched historical data from CoinGecko.")
        return {"prices": data.get('prices', [])}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /shib-historical-data endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process historical data: {e}")

@app.post("/ai-suggest-profit")
async def ai_suggest_profit(request_body: dict):
    print("Received request for /ai-suggest-profit")
    historical_prices = request_body.get("historical_prices", [])
    current_price = request_body.get("current_price")

    if not historical_prices or current_price is None:
        raise HTTPException(status_code=400, detail="Missing historical_prices or current_price in request body.")

    prompt = f"Given the following recent daily closing prices for Shiba Inu (SHIB) in USD: [{', '.join(map(str, historical_prices))}]. The current price is {current_price}. Based on this data, what would be a reasonable and conservative target profit percentage (e.g., 1.5, 2.0, 3.5) for a short-term trade? Provide only the number, no text or percentage sign."
    
    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chat_history}
    
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(gemini_api_url, json=payload, timeout=30.0)
            response.raise_for_status()
            result = response.json()
        print("Successfully called Gemini API.")

        if result.get('candidates') and result['candidates'][0].get('content') and \
           result['candidates'][0]['content'].get('parts') and result['candidates'][0]['content']['parts'][0].get('text'):
            ai_suggestion_text = result['candidates'][0]['content']['parts'][0]['text']
            suggested_profit = float(ai_suggestion_text.strip())
            return {"suggested_profit": suggested_profit}
        else:
            print(f"Gemini API did not return a valid suggestion structure: {result}")
            raise HTTPException(status_code=500, detail="AI did not return a valid suggestion.")
    except ValueError:
        print(f"AI returned non-numeric suggestion: {ai_suggestion_text}")
        raise HTTPException(status_code=500, detail="AI returned a non-numeric suggestion.")
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error calling Gemini API: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Gemini API error: {e.response.text}")
    except httpx.RequestError as e:
        print(f"Network Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Network error calling Gemini API: {e}")
    except Exception as e:
        print(f"Unexpected Error during AI suggestion: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during AI suggestion: {e}")

# New endpoint to save a simulated trade to the database
@app.post("/save-trade")
async def save_trade(trade_data: dict):
    print("Received request for /save-trade")
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection not established.")

    # Ensure numeric values are converted to float for database storage
    try:
        trade_data['entryPrice'] = float(trade_data.get('entryPrice'))
        trade_data['takeProfitPrice'] = float(trade_data.get('takeProfitPrice'))
        trade_data['stopLossPrice'] = float(trade_data.get('stopLossPrice'))
        trade_data['positionSize'] = float(trade_data.get('positionSize'))
        trade_data['profitLoss'] = float(trade_data.get('profitLoss')) if trade_data.get('profitLoss') is not None else None
        trade_data['outcomePrice'] = float(trade_data.get('outcomePrice')) if trade_data.get('outcomePrice') is not None else None
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid numeric data in trade: {e}")

    async with db_pool.acquire() as conn:
        try:
            await conn.execute('''
                INSERT INTO simulated_trades (id, signal_type, entry_price, take_profit_price, stop_loss_price, position_size, status, outcome_price, profit_loss, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    signal_type = EXCLUDED.signal_type,
                    entry_price = EXCLUDED.entry_price,
                    take_profit_price = EXCLUDED.take_profit_price,
                    stop_loss_price = EXCLUDED.stop_loss_price,
                    position_size = EXCLUDED.position_size,
                    status = EXCLUDED.status,
                    outcome_price = EXCLUDED.outcome_price,
                    profit_loss = EXCLUDED.profit_loss,
                    timestamp = EXCLUDED.timestamp;
            ''',
            trade_data['id'], trade_data['signalType'], trade_data['entryPrice'],
            trade_data['takeProfitPrice'], trade_data['stopLossPrice'], trade_data['positionSize'],
            trade_data['status'], trade_data['outcomePrice'], trade_data['profitLoss'],
            trade_data['timestamp']
            )
            print(f"Trade {trade_data['id']} saved/updated in DB.")
            return {"message": "Trade saved successfully"}
        except Exception as e:
            print(f"ERROR: Could not save trade to database: {e}")
            raise HTTPException(status_code=500, detail=f"Database error saving trade: {e}")

# New endpoint to load all simulated trades from the database
@app.get("/get-all-trades")
async def get_all_trades():
    print("Received request for /get-all-trades")
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection not established.")

    async with db_pool.acquire() as conn:
        try:
            records = await conn.fetch('''
                SELECT id, signal_type, entry_price, take_profit_price, stop_loss_price, position_size, status, outcome_price, profit_loss, timestamp
                FROM simulated_trades ORDER BY timestamp DESC;
            ''')
            
            trades = []
            for record in records:
                trade = {
                    "id": record['id'],
                    "signalType": record['signal_type'],
                    "entryPrice": float(record['entry_price']), # Convert Decimal to float
                    "takeProfitPrice": float(record['take_profit_price']),
                    "stopLossPrice": float(record['stop_loss_price']),
                    "positionSize": float(record['position_size']),
                    "status": record['status'],
                    "outcomePrice": float(record['outcome_price']) if record['outcome_price'] is not None else None,
                    "profitLoss": float(record['profit_loss']) if record['profit_loss'] is not None else None,
                    "timestamp": record['timestamp']
                }
                trades.append(trade)
            print(f"Loaded {len(trades)} trades from DB.")
            return {"trades": trades}
        except Exception as e:
            print(f"ERROR: Could not fetch trades from database: {e}")
            raise HTTPException(status_code=500, detail=f"Database error fetching trades: {e}")

# 4. Endpoint to check signal outcome (now updates DB)
@app.post("/check-signal-outcome")
async def check_signal_outcome(trade_details: dict):
    print("Received request for /check-signal-outcome")
    entry_price = trade_details.get("entryPrice")
    take_profit_price = trade_details.get("takeProfitPrice")
    stop_loss_price = trade_details.get("stopLossPrice")
    position_size = trade_details.get("positionSize")
    signal_type = trade_details.get("signalType")
    timestamp = trade_details.get("timestamp")
    trade_id = trade_details.get("id") # Get the ID of the trade to update

    if None in [entry_price, take_profit_price, stop_loss_price, position_size, signal_type, timestamp, trade_id]:
        raise HTTPException(status_code=400, detail="Missing trade details in request body.")

    try:
        from_timestamp_s = timestamp / 1000
        to_timestamp_s = from_timestamp_s + (2 * 24 * 60 * 60)
        
        chart_url = f"https://api.coingecko.com/api/v3/coins/shiba-inu/market_chart/range?vs_currency=usd&from={from_timestamp_s}&to={to_timestamp_s}"
        
        data = await fetch_coingecko_data(chart_url)
        prices = data.get('prices', [])
        print(f"Fetched {len(prices)} historical prices for outcome check.")

        if not prices or len(prices) < 2:
            # Update DB status to pending if not enough data
            if db_pool:
                async with db_pool.acquire() as conn:
                    await conn.execute("UPDATE simulated_trades SET status = $1 WHERE id = $2;", "pending", trade_id)
            return {
                "status": "pending",
                "outcomePrice": entry_price,
                "profitLoss": 0,
                "message": "Not enough historical data to determine outcome within the timeframe."
            }

        outcome = 'pending'
        final_price = entry_price
        profit_loss = 0

        for price_data in prices:
            price = price_data[1]
            final_price = price

            if signal_type == 'LONG':
                if price >= take_profit_price:
                    outcome = 'win'; break
                if price <= stop_loss_price:
                    outcome = 'loss'; break
            else: # SHORT
                if price <= take_profit_price:
                    outcome = 'win'; break
                if price >= stop_loss_price:
                    outcome = 'loss'; break
        
        # Update the trade in the database
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.execute('''
                    UPDATE simulated_trades
                    SET status = $1, outcome_price = $2, profit_loss = $3
                    WHERE id = $4;
                ''', outcome, final_price, profit_loss, trade_id)
            print(f"Trade {trade_id} outcome updated in DB: {outcome}")

        return {
            "status": outcome,
            "outcomePrice": final_price,
            "profitLoss": profit_loss,
            "message": f"Signal outcome: {outcome}"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /check-signal-outcome endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check signal outcome: {e}")

