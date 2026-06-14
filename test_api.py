import requests

data = {
    "temp": 290,
    "rain": 0,
    "snow": 0,
    "clouds": 40
}

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json=data
)

print(response.status_code)
print(response.text)