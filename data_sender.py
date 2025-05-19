import requests
import random
import time

URL = "http://localhost:5001/train"

def generate_data():
    x = random.uniform(0, 10)
    y = 2 * x + 3 + random.gauss(0, 1)
    return {"x": x, "y": y}

while True:
    data = generate_data()
    try:
        res = requests.post(URL, json=data)
        print(f"Gửi: {data} => Phản hồi: {res.text}")
    except Exception as e:
        print("Lỗi:", e)
    time.sleep(1)
