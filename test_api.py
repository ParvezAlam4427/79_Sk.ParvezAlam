import requests
import json

url = "http://localhost:8000/ask"
payload = {"query": "I want to port out"}
headers = {'Content-Type': 'application/json'}

try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(response.json())
except Exception as e:
    print(f"Error: {e}")
