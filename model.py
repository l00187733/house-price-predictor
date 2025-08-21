import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("data.csv")

X = data[["area", "bedrooms", "bathrooms"]]
y = data["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model to disk
with open("house_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as house_model.pkl")
