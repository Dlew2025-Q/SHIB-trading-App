# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import asyncio
import asyncpg
import json
from datetime import datetime, timedelta

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
            # Create table if it doesn't exist, with all expected columns
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

            # Add columns if they don't exist (for existing tables from previous versions)
            try:
                await conn.execute("ALTER TABLE simulated_trades ADD COLUMN IF NOT EXISTS ai_reasoning TEXT;")
                print("Checked/Added 'ai_reasoning' column to 'simulated_trades' table.")
            except Exception as e:
                print(f"WARNING: Could not add 'ai_reasoning' column (might already exist): {e}")

            try:
                await conn.execute("ALTER TABLE simulated_trades ADD COLUMN IF NOT EXISTS sentiment_score NUMERIC(10, 8);")
                print("Checked/Added 'sentiment_score' column to 'simulated_trades' table.")
            except Exception as e:
                print(f"WARNING: Could not add 'sentiment_score' column (might already exist): {e}")

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
    else:
        print("CoinGecko API Key is missing in fetch_coingecko_data. Requests might fail.")

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

# --- Helper function to fetch sentiment data from SentiCrypt ---
async def fetch_senticrypt_data():
    await asyncio.sleep(0.5)
    senticrypt_url = "https://api.senticrypt.com/v2/all.json"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(senticrypt_url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            if data and isinstance(data, list) and len(data) > 0:
                latest_sentiment = data[0]
                return latest_sentiment.get('mean')
            return None
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error fetching SentiCrypt data: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            print(f"Network Error fetching SentiCrypt data: {e}")
            return None
        except Exception as e:
            print(f"Unexpected Error fetching SentiCrypt data: {e}")
            return None

# --- Helper function to fetch crypto news ---
async def fetch_crypto_news(limit: int = 5):
    if not RAPIDAPI_KEY:
        print("ERROR: RAPIDAPI_KEY is not set. Cannot fetch crypto news.")
        return []

    await asyncio.sleep(0.5)
    news_url = f"https://cryptoinfo.p.rapidapi.com/api/private/latest_news/rapid_api/news/{limit}"
    
    headers = {
        'Content-Type': 'application/json',
        'x-rapidapi-host': 'cryptoinfo.p.rapidapi.com',
        'x-rapidapi-key': RAPIDAPI_KEY
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(news_url, headers=headers, timeout=15.0)
            response.raise_for_status()
            data = response.json()
            
            if data and isinstance(data, dict) and 'results' in data and isinstance(data['results'], list):
                return data['results']
            return []
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error fetching Crypto News: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Crypto News API error: {e.response.text}")
        except httpx.RequestError as e:
            print(f"Network Error fetching Crypto News: {e}")
            raise HTTPException(status_code=500, detail=f"Network error fetching Crypto News: {e}")
        except Exception as e:
            print(f"Unexpected Error fetching Crypto News: {e}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred fetching Crypto News: {e}")


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

@app.get("/crypto-news/{limit}")
async def get_crypto_news(limit: int):
    print(f"Received request for /crypto-news/{limit}")
    try:
        news_items = await fetch_crypto_news(limit)
        print(f"Successfully fetched {len(news_items)} news items.")
        return {"news": news_items}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /crypto-news endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch crypto news: {e}")


@app.post("/ai-suggest-profit")
async def ai_suggest_profit(request_body: dict):
    print("Received request for /ai-suggest-profit")
    historical_prices = request_body.get("historical_prices", [])
    current_price = request_body.get("current_price")

    if not historical_prices or current_price is None:
        raise HTTPException(status_code=400, detail="Missing historical_prices or current_price in request body.")

    prompt = f"Given the following recent daily closing prices for Shiba Inu (SHIB) in USD: [{', '.join(map(str, historical_prices))}] and the current price is {current_price}. Based on this data, what would be a reasonable and conservative target profit percentage (e.g., 1.5, 2.0, 3.5) for a short-term trade? Provide only the number, no text or percentage sign. Only give the number."
    
    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chat_history}
    
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "suggested_profit": {"type": "NUMBER"}
                },
                "required": ["suggested_profit"]
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(gemini_api_url, json=payload, timeout=30.0)
            response.raise_for_status()
            result = response.json()
        print("Successfully called Gemini API for profit suggestion.")

        if result.get('candidates') and result['candidates'][0].get('content') and \
           result['candidates'][0]['content'].get('parts') and result['candidates'][0]['content']['parts'][0].get('text'):
            ai_suggestion_text = result['candidates'][0]['content']['parts'][0]['text']
            try:
                parsed_json = json.loads(ai_suggestion_text)
                suggested_profit = parsed_json.get("suggested_profit")
            except json.JSONDecodeError:
                suggested_profit = float(ai_suggestion_text.strip())
            
            if suggested_profit is not None and not isinstance(suggested_profit, bool) and not isinstance(suggested_profit, str) and not isinstance(suggested_profit, list) and not isinstance(suggested_profit, dict):
                return {"suggested_profit": suggested_profit}
            else:
                print(f"AI returned invalid profit suggestion format or value: {ai_suggestion_text}")
                raise HTTPException(status_code=500, detail="AI returned an invalid profit suggestion value.")
        else:
            print(f"Gemini API did not return a valid suggestion structure for profit: {result}")
            raise HTTPException(status_code=500, detail="AI did not return a valid profit suggestion.")
    except ValueError:
        print(f"AI returned non-numeric suggestion for profit after stripping: {ai_suggestion_text}")
        raise HTTPException(status_code=500, detail="AI returned a non-numeric profit suggestion.")
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error calling Gemini API for profit: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Gemini API error for profit: {e.response.text}")
    except httpx.RequestError as e:
        print(f"Network Error calling Gemini API for profit: {e}")
        raise HTTPException(status_code=500, detail=f"Network error calling Gemini API for profit: {e}")
    except Exception as e:
        print(f"Unexpected Error during AI profit suggestion: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during AI profit suggestion: {e}")


# Endpoint for AI-driven trade signal
@app.post("/ai-trade-signal")
async def ai_trade_signal(request_body: dict):
    print("Received request for /ai-trade-signal")
    current_price = request_body.get("current_price")
    price_change_24h = request_body.get("price_change_24h")
    historical_prices = request_body.get("historical_prices", [])
    sentiment_filter_enabled = request_body.get("sentiment_filter_enabled", False)

    if current_price is None or not historical_prices:
        raise HTTPException(status_code=400, detail="Missing current_price or historical_prices for AI signal.")

    current_sentiment_score = None
    if sentiment_filter_enabled:
        current_sentiment_score = await fetch_senticrypt_data()
        if current_sentiment_score is None:
            print("WARNING: Sentiment filter enabled but could not fetch sentiment data. Proceeding without sentiment.")
        else:
            print(f"Fetched current sentiment score: {current_sentiment_score}")

    # --- CONTRARIAN AI PROMPT ---
    prompt_parts = [
        "You are a cautious, contrarian technical analyst for cryptocurrency. Your primary strategy is mean-reversion. You look for opportunities where the price has moved too far, too fast and is likely to correct.",
        "\n--- Core Strategy ---",
        "1. **Identify Extremes:** Your main goal is to identify overbought or oversold conditions. A large 24-hour price change is a strong indicator of an extreme.",
        "2. **Fade the Move:** When you see a strong price surge (e.g., > 5% in 24h), you should strongly consider a SHORT position, anticipating a pullback. Conversely, after a major drop, you look for a LONG opportunity.",
        "3. **Patience is Key:** Do not chase momentum. If the trend is strong but not at an extreme, it is better to be NEUTRAL and wait for a clearer opportunity. Only signal LONG if the price has already pulled back to a support level. Only signal SHORT if the price is hitting a clear resistance level after a sharp rise.",
        
        "\n--- Data Provided ---",
        f"- Current Price: ${current_price}",
        f"- 24-hour Price Change: {price_change_24h:.2f}%",
        f"- Last 30 daily closing prices (oldest to newest): {json.dumps(historical_prices)}",
    ]

    if sentiment_filter_enabled and current_sentiment_score is not None:
        prompt_parts.append(f"- General Market Sentiment (BTC Score): {current_sentiment_score:.4f} (Use this to gauge overall market fear or greed, which can amplify reversals).")
    
    prompt_parts.extend([
        "\n--- Your Analysis ---",
        "Based on your contrarian, mean-reversion strategy and the data provided, provide your response in the following strict JSON format. Your reasoning must be concise and justify your signal based on the strategy.",
        "Example SHORT reasoning: 'The price is up over 8% in 24h, indicating an overbought condition. This rally is likely unsustainable, presenting a mean-reversion SHORT opportunity.'",
        "Example LONG reasoning: 'After a significant drop, the price is now showing signs of stabilizing at a historical support level. This presents a potential LONG opportunity for a bounce.'",
        "Example NEUTRAL reasoning: 'The price is consolidating mid-range with no clear overbought or oversold signal. Waiting for a more extreme price movement before entering a trade.'",
    ])
    
    prompt = "\n".join(prompt_parts)

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chat_history}
    
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "signal_type": {"type": "STRING", "enum": ["LONG", "SHORT", "NEUTRAL"]},
                    "reasoning": {"type": "STRING"}
                },
                "required": ["signal_type", "reasoning"]
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(gemini_api_url, json=payload, timeout=45.0)
            response.raise_for_status()
            result = response.json()
        print("Successfully called Gemini API for trade signal.")

        if result.get('candidates') and result['candidates'][0].get('content') and \
           result['candidates'][0]['content'].get('parts') and result['candidates'][0]['content']['parts'][0].get('text'):
            ai_response_json_str = result['candidates'][0]['content']['parts'][0]['text']
            ai_response = json.loads(ai_response_json_str)

            signal_type = ai_response.get("signal_type")
            reasoning = ai_response.get("reasoning")

            if signal_type in ["LONG", "SHORT", "NEUTRAL"] and reasoning:
                return {"signal_type": signal_type, "reasoning": reasoning, "sentiment_score": current_sentiment_score}
            else:
                print(f"AI returned invalid signal format: {ai_response_json_str}")
                raise HTTPException(status_code=500, detail="AI returned invalid signal format.")
        else:
            print(f"Gemini API did not return a valid signal structure: {result}")
            raise HTTPException(status_code=500, detail="AI did not return a valid signal.")
    except json.JSONDecodeError as e:
        print(f"Failed to parse AI response JSON: {e} - Raw: {ai_response_json_str}")
        raise HTTPException(status_code=500, detail=f"AI response format error: {e}")
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error calling Gemini API for signal: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Gemini API error for signal: {e.response.text}")
    except httpx.RequestError as e:
        print(f"Network Error calling Gemini API for signal: {e}")
        raise HTTPException(status_code=500, detail=f"Network error calling Gemini API for signal: {e}")
    except Exception as e:
        print(f"Unexpected Error during AI trade signal generation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during AI trade signal generation: {e}")

@app.post("/save-trade")
async def save_trade(trade_data: dict):
    print("Received request for /save-trade")
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection not established.")

    try:
        # Ensure numeric values are converted to float for database storage
        trade_data['entryPrice'] = float(trade_data.get('entryPrice'))
        trade_data['takeProfitPrice'] = float(trade_data.get('takeProfitPrice'))
        trade_data['stopLossPrice'] = float(trade_data.get('stopLossPrice'))
        trade_data['positionSize'] = float(trade_data.get('positionSize'))
        trade_data['profitLoss'] = float(trade_data.get('profitLoss')) if trade_data.get('profitLoss') is not None else None
        trade_data['outcomePrice'] = float(trade_data.get('outcomePrice')) if trade_data.get('outcomePrice') is not None else None
        trade_data['ai_reasoning'] = trade_data.get('aiReasoning', '')
        trade_data['sentiment_score'] = float(trade_data.get('sentimentScore')) if trade_data.get('sentimentScore') is not None else None
    except (ValueError, TypeError) as e:
        print(f"ERROR: Invalid numeric data in trade for /save-trade: {e} - Data: {trade_data}")
        raise HTTPException(status_code=400, detail=f"Invalid numeric data in trade: {e}")

    async with db_pool.acquire() as conn:
        try:
            await conn.execute('''
                INSERT INTO simulated_trades (id, signal_type, entry_price, take_profit_price, stop_loss_price, position_size, status, outcome_price, profit_loss, timestamp, ai_reasoning, sentiment_score)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (id) DO UPDATE SET
                    signal_type = EXCLUDED.signal_type,
                    entry_price = EXCLUDED.entry_price,
                    take_profit_price = EXCLUDED.take_profit_price,
                    stop_loss_price = EXCLUDED.stop_loss_price,
                    position_size = EXCLUDED.position_size,
                    status = EXCLUDED.status,
                    outcome_price = EXCLUDED.outcome_price,
                    profit_loss = EXCLUDED.profit_loss,
                    timestamp = EXCLUDED.timestamp,
                    ai_reasoning = EXCLUDED.ai_reasoning,
                    sentiment_score = EXCLUDED.sentiment_score;
            ''',
            trade_data['id'], trade_data['signalType'], trade_data['entryPrice'],
            trade_data['takeProfitPrice'], trade_data['stopLossPrice'], trade_data['positionSize'],
            trade_data['status'], trade_data['outcomePrice'], trade_data['profitLoss'],
            trade_data['timestamp'], trade_data['ai_reasoning'], trade_data['sentiment_score']
            )
            print(f"Trade {trade_data['id']} saved/updated in DB.")
            return {"message": "Trade saved successfully"}
        except Exception as e:
            print(f"ERROR: Could not save trade to database: {e}")
            raise HTTPException(status_code=500, detail=f"Database error saving trade: {e}")

@app.get("/get-all-trades")
async def get_all_trades():
    print("Received request for /get-all-trades")
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection not established.")

    async with db_pool.acquire() as conn:
        try:
            records = await conn.fetch('''
                SELECT id, signal_type, entry_price, take_profit_price, stop_loss_price, position_size, status, outcome_price, profit_loss, timestamp, ai_reasoning, sentiment_score
                FROM simulated_trades ORDER BY timestamp DESC;
            ''')
            
            trades = []
            for record in records:
                trade = {
                    "id": record['id'],
                    "signalType": record['signal_type'],
                    "entryPrice": float(record['entry_price']),
                    "takeProfitPrice": float(record['take_profit_price']),
                    "stopLossPrice": float(record['stop_loss_price']),
                    "positionSize": float(record['position_size']),
                    "status": record['status'],
                    "outcomePrice": float(record['outcome_price']) if record['outcome_price'] is not None else None,
                    "profitLoss": float(record['profit_loss']) if record['profit_loss'] is not None else None,
                    "timestamp": record['timestamp'],
                    "aiReasoning": record['ai_reasoning'],
                    "sentimentScore": float(record['sentiment_score']) if record['sentiment_score'] is not None else None
                }
                trades.append(trade)
            print(f"Loaded {len(trades)} trades from DB.")
            return {"trades": trades}
        except Exception as e:
            print(f"ERROR: Could not fetch trades from database: {e}")
            raise HTTPException(status_code=500, detail=f"Database error fetching trades: {e}")

@app.post("/check-signal-outcome")
async def check_signal_outcome(trade_details: dict):
    print("Received request for /check-signal-outcome")
    entry_price = trade_details.get("entryPrice")
    take_profit_price = trade_details.get("takeProfitPrice")
    stop_loss_price = trade_details.get("stopLossPrice")
    position_size = trade_details.get("positionSize")
    signal_type = trade_details.get("signalType")
    timestamp = trade_details.get("timestamp")
    trade_id = trade_details.get("id")

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
                    outcome = 'win'
                    break
                if price <= stop_loss_price:
                    outcome = 'loss'
                    break
            else: # SHORT
                if price <= take_profit_price:
                    outcome = 'win'
                    break
                if price >= stop_loss_price:
                    outcome = 'loss'
                    break
        
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
