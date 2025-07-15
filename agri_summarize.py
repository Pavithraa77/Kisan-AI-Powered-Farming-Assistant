from google.adk.agents import Agent
import requests
import datetime
import os
from typing import Dict




API_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
API_KEY = os.environ.get("API_KEY")


def get_market_prices(state: str = "Tamil Nadu", district: str = "Coimbatore",
                      market: str = "Coimbatore", commodity: str = "Tomato",
                      variety: str = "Local", grade: str = "A") -> dict:
    try:
        params = {
            "api-key": API_KEY,
            "format": "json",
            "limit": 10,
            "filters[state]": state,
            "filters[district]": district,
            "filters[market]": market,
            "filters[commodity]": commodity,
            "filters[variety]": variety,
            "filters[grade]": grade
        }

        response = requests.get(API_URL, params=params, timeout=10)
        print("Market Price URL:", response.url)
        print(" Raw Response (truncated):", response.text[:300])
        response.raise_for_status()

        data = response.json()
        if not data.get("records"):
            return {"status": "error", "error_message": "No market price data found."}

        return {
            "status": "success",
            "data": data["records"],
            "summary": f"Found {len(data['records'])} price records for {commodity}"
        }

    except requests.exceptions.RequestException as e:
        return {"status": "error", "error_message": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"status": "error", "error_message": f"Unexpected error: {str(e)}"}


def get_commodity_arrival_data(state: str = "Tamil Nadu", district: str = "Coimbatore",
                                commodity: str = "Tomato", arrival_date: str = None) -> dict:
    try:
        if arrival_date is None:
            arrival_date = datetime.datetime.now().strftime("%d/%m/%Y")  # Match API format

        params = {
            "api-key": API_KEY,
            "format": "json",
            "limit": 10,
            "filters[state]": state,
            "filters[district]": district,
            "filters[commodity]": commodity,
            "filters[arrival_date]": arrival_date
        }

        response = requests.get(API_URL, params=params, timeout=10)
        print("Arrival Data URL:", response.url)
        print("Raw Response (truncated):", response.text[:300])
        response.raise_for_status()

        data = response.json()
        if not data.get("records"):
            return {"status": "error", "error_message": "No arrival data found."}

        return {
            "status": "success",
            "data": data["records"],
            "summary": f"Arrival data for {commodity} in {district}, {state} on {arrival_date}"
        }

    except requests.exceptions.RequestException as e:
        return {"status": "error", "error_message": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"status": "error", "error_message": f"Unexpected error: {str(e)}"}


def analyze_market_trends(commodity: str = "Tomato", state: str = "Tamil Nadu",
                          district: str = "Coimbatore") -> dict:
    try:
        price_data = get_market_prices(state=state, district=district, commodity=commodity)
        if price_data["status"] != "success":
            return {"status": "error", "error_message": price_data["error_message"]}

        records = price_data["data"]
        prices = []

        for record in records:
            try:
                price = float(record.get("modal_price", 0))
                prices.append(price)
            except:
                continue

        if not prices:
            return {"status": "error", "error_message": "No valid price data to analyze."}

        avg_price = sum(prices) / len(prices)
        max_price = max(prices)
        min_price = min(prices)

        insights = [
            f" Average modal price for {commodity}: ₹{avg_price:.2f}",
            f"Price range: ₹{min_price:.2f} – ₹{max_price:.2f}",
        ]

        variance = max_price - min_price
        if variance > 0.2 * avg_price:
            insights.append(" High price volatility detected.")
        else:
            insights.append("Prices appear stable.")

        trend = "Trending lower" if avg_price < (min_price + max_price) / 2 else "Trending higher"
        insights.append(trend)

        insights.append(f"Based on {len(prices)} records from {district}, {state}")
        insights.append(f"Analysis on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            "status": "success",
            "insights": insights,
            "raw_data": records,
            "summary": f"Market trend analysis for {commodity}"
        }

    except Exception as e:
        return {"status": "error", "error_message": f"Trend analysis error: {str(e)}"}



agri_agent = Agent(
    name="agricultural_market_agent",
    model="gemini-2.0-flash",
    description="Agent that provides agricultural market prices, arrivals, and insights using Indian government data.",
    instruction="""
I can help you with:
- Current market prices for agricultural commodities across India
- Arrival data of commodities in specific mandis
- Price trend analysis and market volatility reports
""",
    tools=[get_market_prices, get_commodity_arrival_data, analyze_market_trends]
)
