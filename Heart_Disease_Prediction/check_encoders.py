import joblib

encoders = joblib.load("encoders.pkl")

for column, encoder in encoders.items():
    print(column, ":", list(encoder.classes_))