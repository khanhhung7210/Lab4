from flask import Flask, request, jsonify
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

app = Flask(__name__)
model = SGDRegressor()
model_initialized = False
X_train, y_train = [], []

@app.route("/train", methods=["POST"])
def train():
    global model_initialized, X_train, y_train

    data = request.get_json()
    x = data["x"]
    y = data["y"]

    X_train.append([x])
    y_train.append(y)

    if len(X_train) >= 10:
        X_np = np.array(X_train)
        y_np = np.array(y_train)

        if not model_initialized:
            model.partial_fit(X_np, y_np)
            model_initialized = True
        else:
            model.partial_fit(X_np[-5:], y_np[-5:])  # chỉ huấn luyện batch mới

        y_pred = model.predict(X_np)
        rmse = mean_squared_error(y_np, y_pred, squared=False)

        return jsonify({
            "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "rmse": rmse,
            "samples": len(X_train)
        })

    return f"Đã nhận {len(X_train)} mẫu, cần ít nhất 10 để huấn luyện.", 200

if __name__ == "__main__":
    app.run(port=5001)
