# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx # Used to make web requests from your backend (like fetching data from CoinGecko/Gemini)
import os    # Used to read secret keys and URLs safely from Render's environment
import asyncio # Import asyncio for sleep

app = FastAPI()

# This is SUPER IMPORTANT for web apps!
# It tells your backend that your frontend (which will be on a different web address)
# is allowed to talk to it. Without this, your browser will block the communication.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In a real, secure app, you'd put your frontend's exact URL here (e.g., "https://your-frontend.render.com")
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Get your CoinGecko API key from Render's environment variables
# You will set COINGECKO_API_KEY in Render later
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
if not COINGECKO_API_KEY:
    print("WARNING: COINGECKO_API_KEY environment variable is not set!")

# The Gemini API key is automatically provided by Canvas for specific models.
# If you were deploying this completely outside Canvas and needed a Gemini API key
# for a paid tier or specific model, you'd set it as an environment variable too.
# For now, we'll use an empty string as instructed for Canvas's automatic injection.
GEMINI_API_KEY = "" # Render's environment might also provide this if configured

# --- Helper function to make CoinGecko API calls ---
async def fetch_coingecko_data(url: str):
    # Add a small delay to be polite to CoinGecko's API and avoid rate limits
    await asyncio.sleep(0.5) # Wait for 0.5 seconds

    headers = {}
    if COINGECKO_API_KEY:
        headers["x-cg-demo-api-key"] = COINGECKO_API_KEY # Add your API key if available
    else:
        print("CoinGecko API Key is missing. Proceeding without it, but requests might fail.")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=10.0) # Add a timeout
            response.raise_for_status() # This will raise an error for 4xx/5xx responses
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

# --- Backend Endpoints (Your Backend's Menu Items) ---

# 0. Root Endpoint (for health checks and direct visits)
@app.get("/")
async def read_root():
    return {"message": "Welcome to the SHIB Trading Analysis Backend API!"}

# 1. Endpoint to get current SHIB prices and market stats
@app.get("/shib-prices")
async def get_shib_prices():
    """
    Fetches current market data for Shiba Inu from CoinGecko.
    This replaces the direct CoinGecko call from the frontend.
    """
    print("Received request for /shib-prices")
    market_data_url = 'https://api.coingecko.com/api/v3/coins/shiba-inu?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false'
    
    try:
        data = await fetch_coingecko_data(market_data_url)
        market_data = data.get('market_data', {})
        print("Successfully fetched market data from CoinGecko.")

        # Extract and return only the data points the frontend needs
        return {
            "current_price": market_data.get('current_price', {}).get('usd'),
            "price_change_24h": market_data.get('price_change_percentage_24h'),
            "market_cap": market_data.get('market_cap', {}).get('usd'),
            "total_volume": market_data.get('total_volume', {}).get('usd'),
            "circulating_supply": market_data.get('circulating_supply'),
            "ath": market_data.get('ath', {}).get('usd')
        }
    except HTTPException:
        raise # Re-raise the HTTPException
    except Exception as e:
        print(f"Error in /shib-prices endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process SHIB prices: {e}")

# 2. Endpoint to get historical data for AI analysis and outcome checks
@app.get("/shib-historical-data")
async def get_shib_historical_data():
    """
    Fetches historical market chart data for Shiba Inu (last 30 days) from CoinGecko.
    This data is used by the frontend for AI profit suggestion and signal outcome checks.
    """
    print("Received request for /shib-historical-data")
    chart_url = f"https://api.coingecko.com/api/v3/coins/shiba-inu/market_chart?vs_currency=usd&days=30"
    
    try:
        data = await fetch_coingecko_data(chart_url)
        print("Successfully fetched historical data from CoinGecko.")
        return {"prices": data.get('prices', [])} # Return array of [timestamp, price]
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /shib-historical-data endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process historical data: {e}")

# 3. Endpoint for AI profit suggestion
@app.post("/ai-suggest-profit")
async def ai_suggest_profit(request_body: dict):
    """
    Sends historical price data to the Gemini API to get a suggested profit percentage.
    """
    print("Received request for /ai-suggest-profit")
    historical_prices = request_body.get("historical_prices", [])
    current_price = request_body.get("current_price")

    if not historical_prices or current_price is None:
        raise HTTPException(status_code=400, detail="Missing historical_prices or current_price in request body.")

    prompt = f"Given the following recent daily closing prices for Shiba Inu (SHIB) in USD: [{', '.join(map(str, historical_prices))}]. The current price is {current_price}. Based on this data, what would be a reasonable and conservative target profit percentage (e.g., 1.5, 2.0, 3.5) for a short-term trade? Provide only the number, no text or percentage sign."
    
    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chat_history}
    
    # Use the GEMINI_API_KEY if available, otherwise it will be empty for Canvas's auto-injection
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(gemini_api_url, json=payload, timeout=30.0) # Increased timeout for AI
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

# 4. Endpoint to check signal outcome (simulated using historical data)
@app.post("/check-signal-outcome")
async def check_signal_outcome(trade_details: dict):
    """
    Simulates checking the outcome of a trade signal against historical data.
    """
    print("Received request for /check-signal-outcome")
    entry_price = trade_details.get("entryPrice")
    take_profit_price = trade_details.get("takeProfitPrice")
    stop_loss_price = trade_details.get("stopLossPrice")
    position_size = trade_details.get("positionSize")
    signal_type = trade_details.get("signalType")
    timestamp = trade_details.get("timestamp") # This is in milliseconds from frontend

    if None in [entry_price, take_profit_price, stop_loss_price, position_size, signal_type, timestamp]:
        raise HTTPException(status_code=400, detail="Missing trade details in request body.")

    try:
        # Fetch historical data for a period after the signal was generated
        # For simplicity, let's fetch data for the next 2 days from the signal timestamp
        from_timestamp_s = timestamp / 1000 # Convert ms to seconds
        to_timestamp_s = from_timestamp_s + (2 * 24 * 60 * 60) # 2 days later in seconds
        
        chart_url = f"https://api.coingecko.com/api/v3/coins/shiba-inu/market_chart/range?vs_currency=usd&from={from_timestamp_s}&to={to_timestamp_s}"
        
        data = await fetch_coingecko_data(chart_url)
        prices = data.get('prices', [])
        print(f"Fetched {len(prices)} historical prices for outcome check.")

        if not prices or len(prices) < 2: # Need at least 2 data points to see movement
            return {
                "status": "pending",
                "outcomePrice": entry_price, # Default to entry price if no data
                "profitLoss": 0,
                "message": "Not enough historical data to determine outcome within the timeframe."
            }

        outcome = 'pending'
        final_price = entry_price # Default to entry if no movement
        profit_loss = 0

        # Iterate through the prices after the entry to check for TP/SL hit
        # Start from the first price after the signal's timestamp
        for price_data in prices:
            price = price_data[1]
            final_price = price # Keep track of the last price seen

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
        
        print(f"Signal outcome determined: {outcome}")
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
