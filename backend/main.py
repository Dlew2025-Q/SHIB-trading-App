# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Important for web apps!
import httpx # To make requests to CoinGecko/Gemini
import os # To read secret keys safely

app = FastAPI()

# This is super important! It tells your backend that your frontend (from a different web address)
# is allowed to talk to it. Without this, your browser will block the communication.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In a real app, you'd put your frontend's exact URL here (e.g., "https://your-frontend.render.com")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# We'll get the CoinGecko API key from a secret place later (environment variables)
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # This will be automatically provided by Canvas for Gemini API calls

# This is our first "menu item" or endpoint
@app.get("/shib-prices")
async def get_shib_prices():
    # This is where the CoinGecko market data fetch logic from your JavaScript goes
    # It will use httpx instead of fetch
    market_data_url = 'https://api.coingecko.com/api/v3/coins/shiba-inu?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false&sparkline=false'
    headers = {"x-cg-demo-api-key": COINGECKO_API_KEY} # Add your CoinGecko API key here
    async with httpx.AsyncClient() as client:
        response = await client.get(market_data_url, headers=headers)
        response.raise_for_status() # Check for errors
        data = response.json()

    # You'll process the data here and return what the frontend needs
    return {"current_price": data['market_data']['current_price']['usd'],
            "price_change_24h": data['market_data']['price_change_percentage_24h'],
            "market_cap": data['market_data']['market_cap']['usd'],
            "total_volume": data['market_data']['total_volume']['usd'],
            "circulating_supply": data['market_data']['circulating_supply'],
            "ath": data['market_data']['ath']['usd']}

# This is another "menu item" for AI profit suggestion
@app.post("/ai-suggest-profit")
async def ai_suggest_profit(historical_prices: list[float]):
    # This is where the Gemini API call logic from your JavaScript goes
    # It will use httpx instead of fetch
    prompt = f"Given the following recent daily closing prices for Shiba Inu (SHIB) in USD: {historical_prices}. Based on this data, what would be a reasonable and conservative target profit percentage (e.g., 1.5, 2.0, 3.5) for a short-term trade? Provide only the number, no text or percentage sign."

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chat_history}

    gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    async with httpx.AsyncClient() as client:
        response = await client.post(gemini_api_url, headers={"Content-Type": "application/json"}, json=payload, params={"key": GEMINI_API_KEY})
        response.raise_for_status()
        result = response.json()

    # Process Gemini's response and return the number
    ai_suggestion_text = result["candidates"][0]["content"]["parts"][0]["text"]
    suggested_profit = float(ai_suggestion_text.strip())
    return {"suggested_profit": suggested_profit}

# You'd add more endpoints for checking signal outcome, etc.