import joblib
import pandas as pd

model = joblib.load('model/client_classifier_model.pkl')

X = pd.DataFrame([{
    'invoice_count': 50,
    'avg_amount': 1000.0,
    'avg_remaining': 0.0,  # No remaining debt
    'avg_delay': 5,
    'on_time_percentage': 95,
    'very_late_payments': 50,
}])

print(model.predict(X)[0])
