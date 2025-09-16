import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load training data
print("Loading training data...")
df = pd.read_csv('data/invoice.csv')

# Convert dates
df['create_date'] = pd.to_datetime(df['create_date'])
df['due_in_date'] = pd.to_datetime(df['due_in_date'])
df['clear_date'] = pd.to_datetime(df['clear_date'])

# Calculate payment delay
df['payment_delay'] = (df['clear_date'] - df['due_in_date']).dt.days

# Create client features
print("Creating client features...")
client_data = []
for client in df['nom_du_client'].unique():
    client_invoices = df[df['nom_du_client'] == client]
    
    # Invoice count SUM
    invoice_count = len(client_invoices) 
    # average montant totale of each clent
    avg_amount = client_invoices['montant_totale'].mean()
    # average totale reste for each clietn
    avg_remaining = client_invoices['totale_reste'].mean()
    # average payement delay in days
    avg_delay = client_invoices['payment_delay'].mean()
    
    on_time_payments = len(client_invoices[client_invoices['payment_delay'] <= 0])
    total_payments = len(client_invoices[client_invoices['payment_delay'].notna()])
    on_time_percentage = (on_time_payments / total_payments * 100) if total_payments > 0 else 0
    very_late_payments = len(client_invoices[client_invoices['payment_delay'] > 45])
    
    client_data.append({
        'client_name': client,
        'invoice_count': invoice_count,
        'avg_amount': avg_amount,
        'avg_remaining': avg_remaining,
        'avg_delay': avg_delay,
        'on_time_percentage': on_time_percentage,
        'very_late_payments': very_late_payments
    })

clients_df = pd.DataFrame(client_data)

# Calculate global averages
global_avg_invoices = clients_df['invoice_count'].mean()
global_avg_amount = df['montant_totale'].mean()

# Persist thresholds used for labeling so test/inference doesn't read training CSV
thresholds = {
    "high_invoice_threshold": float(global_avg_invoices * 0.8 / 0.5),
    "mid_low_invoice_threshold": float(global_avg_invoices * 0.2 / 0.5),
    "high_rest_threshold": float(global_avg_amount * 0.75 / 0.5),
    "mid_rest_low_threshold": float(global_avg_amount * 0.25 / 0.5),
    "mid_rest_high_threshold": float(global_avg_amount * 0.75 / 0.5)
}

# Create labels
print("Creating labels...")
labels = []
for _, client in clients_df.iterrows():
    invoice_count = client['invoice_count']
    avg_remaining = client['avg_remaining']
    on_time_percentage = client['on_time_percentage']
    very_late = client['very_late_payments']
    avg_delay = client['avg_delay']
    

    # Precompute scaled thresholds using the provided "/ 0.5" rule
    # Default label to avoid NameError when none of the conditions match
  
    high_invoice_threshold = global_avg_invoices * 0.8 / 0.5  # 1.6 * avg
    mid_low_invoice_threshold = global_avg_invoices * 0.2 / 0.5  # 0.4 * avg
    high_rest_threshold = global_avg_amount * 0.75 / 0.5  # 1.5 * avg amount
    mid_rest_low_threshold = global_avg_amount * 0.25 / 0.5  # 0.5 * avg amount
    mid_rest_high_threshold = global_avg_amount * 0.75 / 0.5  # 1.5 * avg amount
    
    if on_time_percentage == 100:
        label = 'EXCELLENT_CLIENT'
    elif (on_time_percentage >= 85 and on_time_percentage < 100 ):
        label = 'GOOD_CLIENT'
    elif  very_late == 0 and avg_delay > 0 and avg_delay <= 45:
        # AVERAGE: a bit late 1-45 days, no payments later than 45 days
        label = 'AVERAGE_CLIENT'
    elif (
        invoice_count > high_invoice_threshold and
        avg_remaining > high_rest_threshold and
        very_late > 0
    ):
        # 3rd degree bad client
        label = '3RD_DEGREE_BAD_CLIENT'
    elif (
        # 2nd degree bad client - condition set 1
        (invoice_count > mid_low_invoice_threshold and invoice_count <= high_invoice_threshold and
         avg_remaining > high_rest_threshold and
         very_late > 0)
        or
        # 2nd degree bad client - condition set 2
        (invoice_count > high_invoice_threshold and
         avg_remaining > mid_rest_low_threshold and avg_remaining <= mid_rest_high_threshold and
         very_late > 0)
    ):
        label = '2ND_DEGREE_BAD_CLIENT'
    elif (
        # 1st degree bad client
        (invoice_count > mid_low_invoice_threshold and invoice_count <= high_invoice_threshold and
         avg_remaining > mid_rest_low_threshold and avg_remaining < high_rest_threshold and
         very_late > 0)
    ):
        label = '1ST_DEGREE_BAD_CLIENT'


    labels.append(label)

clients_df['label'] = labels

# Prepare features
features = ['invoice_count', 'avg_amount', 'avg_remaining', 'avg_delay', 'on_time_percentage', 'very_late_payments']
X = clients_df[features]
y = clients_df['label']

# Handle missing values to avoid NaNs during training
X = X.fillna(0)

# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Test on validation set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Validation Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'model/client_classifier_model.pkl')
print("Model saved to: model/client_classifier_model.pkl")

# Save results
clients_df.to_csv('data/training_results.csv', index=False)
print("Training results saved to: data/training_results.csv")

print("\nTraining completed!")
print("Label distribution:")
print(clients_df['label'].value_counts())
