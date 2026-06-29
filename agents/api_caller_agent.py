import os
import json
import requests
from groq import Groq
from dotenv import load_dotenv
from utils.cosmos_logger import log_event

load_dotenv()

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name e.g. Mumbai"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock",
            "description": "Get current stock price for a ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker e.g. AAPL, TSLA"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_crypto",
            "description": "Get current price of a cryptocurrency",
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {"type": "string", "description": "Coin id e.g. bitcoin, ethereum, solana"}
                },
                "required": ["coin"]
            }
        }
    }
]

def _get_weather(city):
    try:
        r = requests.get(f"https://wttr.in/{city}?format=j1", timeout=8)
        data = r.json()
        current = data["current_condition"][0]
        return {
            "city": city,
            "temp_c": current["temp_C"],
            "feels_like_c": current["FeelsLikeC"],
            "description": current["weatherDesc"][0]["value"],
            "humidity": current["humidity"]
        }
    except Exception as e:
        return {"error": str(e)}

def _get_stock(ticker):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=8)
        data = r.json()
        meta = data["chart"]["result"][0]["meta"]
        return {
            "ticker": ticker,
            "price": meta.get("regularMarketPrice"),
            "currency": meta.get("currency"),
            "exchange": meta.get("exchangeName")
        }
    except Exception as e:
        return {"error": str(e)}

def _get_crypto(coin):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd,inr"
        r = requests.get(url, timeout=8)
        data = r.json()
        if coin not in data:
            return {"error": f"Coin '{coin}' not found"}
        return {"coin": coin, "usd": data[coin]["usd"], "inr": data[coin]["inr"]}
    except Exception as e:
        return {"error": str(e)}

_tool_map = {
    "get_weather": _get_weather,
    "get_stock": _get_stock,
    "get_crypto": _get_crypto
}

def run(task, workflow_id="standalone"):
    messages = [
        {"role": "system", "content": "You are a real-time data assistant. Use the available tools to fetch accurate live data. Always call a tool when data is needed."},
        {"role": "user", "content": task}
    ]
    for _ in range(5):
        response = _groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=_tools,
            tool_choice="auto",
            max_tokens=1024
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]})
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                result = _tool_map[fn_name](**fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)
                })
        else:
            answer = msg.content
            log_event(workflow_id, "APICallerAgent", task, answer)
            return answer
    answer = messages[-1].get("content", "Could not complete the API task.")
    log_event(workflow_id, "APICallerAgent", task, answer)
    return answer
