import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
print("Loading model...")
model = joblib.load('model/client_classifier_model.pkl')

# Load test data
print("Loading test data...")
df = pd.read_csv('data/invoices_v2.csv')

# Convert dates
df['create_date'] = pd.to_datetime(df['create_date'])
df['due_in_date'] = pd.to_datetime(df['due_in_date'])
df['clear_date'] = pd.to_datetime(df['clear_date'])

# Calculate payment delay
df['payment_delay'] = (df['clear_date'] - df['due_in_date']).dt.days

# Create client features (same as training)
print("Creating client features...")
client_data = []
for client in df['nom_du_client'].unique():
    client_invoices = df[df['nom_du_client'] == client]
    
    invoice_count = len(client_invoices)
    avg_amount = client_invoices['montant_totale'].mean()
    avg_remaining = client_invoices['totale_reste'].mean()
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

# Load training data to get global averages (same as used in training)
df_train = pd.read_csv('data/invoices.csv')
clients_train_df = pd.DataFrame()
for client in df_train['nom_du_client'].unique():
    client_invoices = df_train[df_train['nom_du_client'] == client]
    clients_train_df = pd.concat([clients_train_df, pd.DataFrame([{
        'invoice_count': len(client_invoices)
    }])], ignore_index=True)

global_avg_invoices = clients_train_df['invoice_count'].mean()
global_avg_amount = df_train['montant_totale'].mean()

# Create labels (same logic as training)
print("Creating labels...")
labels = []
for _, client in clients_df.iterrows():
    invoice_count = client['invoice_count']
    avg_remaining = client['avg_remaining']
    on_time_percentage = client['on_time_percentage']
    very_late = client['very_late_payments']
    avg_delay = client['avg_delay']
    
    if on_time_percentage == 100:
        label = 'EXCELLENT_CLIENT'
    elif on_time_percentage >= 90:
        label = 'GOOD_CLIENT'
    elif very_late == 0 and avg_delay > 0 and avg_delay <= 45:
        label = 'AVERAGE_CLIENT'
    elif (invoice_count > global_avg_invoices * 0.2 and 
          invoice_count <= global_avg_invoices * 0.8 and 
          avg_remaining > global_avg_amount * 0.25 and 
          avg_remaining <= global_avg_amount * 0.75 and 
          very_late > 0):
        label = '1ST_DEGREE_BAD_CLIENT'
    elif (invoice_count > global_avg_invoices * 0.2 and 
          invoice_count <= global_avg_invoices * 0.8 and 
          avg_remaining > global_avg_amount * 0.75 and 
          very_late > 0):
        label = '2ND_DEGREE_BAD_CLIENT'
    elif (invoice_count > global_avg_invoices * 0.8 and 
          avg_remaining > global_avg_amount * 0.25 and 
          avg_remaining <= global_avg_amount * 0.75 and 
          very_late > 0):
        label = '2ND_DEGREE_BAD_CLIENT'
    elif (invoice_count > global_avg_invoices * 0.8 and 
          avg_remaining > global_avg_amount * 0.75 and 
          very_late > 0):
        label = '3RD_DEGREE_BAD_CLIENT'
    else:
        if very_late > 0:
            label = '1ST_DEGREE_BAD_CLIENT'
        else:
            label = 'AVERAGE_CLIENT'
    
    labels.append(label)

clients_df['label'] = labels

# Prepare features
features = ['invoice_count', 'avg_amount', 'avg_remaining', 'avg_delay', 'on_time_percentage', 'very_late_payments']
X = clients_df[features]
y = clients_df['label']

# Make predictions
print("Making predictions...")
y_pred = model.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Test Accuracy: {accuracy:.3f}")

# Show classification report
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Save results
clients_df['predicted_label'] = y_pred
clients_df.to_csv('data/test_results.csv', index=False)
print("Test results saved to: data/test_results.csv")

print("\nActual vs Predicted:")
for i, row in clients_df.iterrows():
    print(f"{row['client_name']}: {row['label']} -> {row['predicted_label']}")
