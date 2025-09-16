import joblib
import pandas as pd

model = joblib.load(r'c:\Users\UNITECH\Desktop\webs\Testing\invoices-Ai-Colab\Client-Invoices-Classification-ML-\model\client_classifier_model.pkl')

X = pd.DataFrame([{
    'invoice_count': 10,
    'avg_amount': 1000.0,
    'avg_remaining': 0.0,
    'avg_delay': 12.5,
    'on_time_percentage': 92,
    'very_late_payments': 0,
    'unpaid_overdue': 0 ,
    'total_unpaid': 0,                   
}])

print(model.predict(X)[0])