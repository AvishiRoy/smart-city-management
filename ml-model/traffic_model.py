import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("dataset/Metro_Interstate_Traffic_Volume.csv")

df = df.dropna()

X = df[['temp','rain_1h','snow_1h','clouds_all']]
y = df['traffic_volume']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)

joblib.dump(model, "ml-model/traffic_model.pkl")

print("Model saved successfully!")