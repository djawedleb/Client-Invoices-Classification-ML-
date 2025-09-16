import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the data
print("Loading data...")
df = pd.read_csv('/home/djawed/Desktop/Ai_Training/data/invoices_dataset.csv')

# Step 2: Create client features
print("Creating client features...")

# Convert dates
df['create_date'] = pd.to_datetime(df['create_date'])
df['due_in_date'] = pd.to_datetime(df['due_in_date'])
df['clear_date'] = pd.to_datetime(df['clear_date'])

# Calculate payment delay
df['payment_delay'] = (df['clear_date'] - df['due_in_date']).dt.days

# Group by client and calculate features
client_data = []
for client in df['nom_du_client'].unique():
    client_invoices = df[df['nom_du_client'] == client]
    
    # Basic features
    invoice_count = len(client_invoices)
    avg_amount = client_invoices['montant_totale'].mean()
    avg_remaining = client_invoices['totale_reste'].mean()
    avg_delay = client_invoices['payment_delay'].mean()
    
    # Payment behavior
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

# Step 3: Create labels based on rules
print("Creating labels...")

# Calculate global averages
global_avg_invoices = clients_df['invoice_count'].mean()
global_avg_amount = df['montant_totale'].mean()

print(f"Global avg invoices: {global_avg_invoices:.2f}")
print(f"Global avg amount: {global_avg_amount:.2f}")

labels = []
for _, client in clients_df.iterrows():
    invoice_count = client['invoice_count']
    avg_remaining = client['avg_remaining']
    on_time_percentage = client['on_time_percentage']
    very_late = client['very_late_payments']
    avg_delay = client['avg_delay']
    
    # Classification rules (EXACTLY following your specifications)
    
    # 6. EXCELLENT CLIENT: 100% on-time payments
    if on_time_percentage == 100:
        label = 'EXCELLENT_CLIENT'
    
    # 5. GOOD CLIENT: 90%+ on-time payments (but not 100%)
    elif on_time_percentage >= 90 and on_time_percentage < 100:
        label = 'GOOD_CLIENT'
    
    # 4. AVERAGE CLIENT: 1-45 days late (no very late payments)
    elif very_late == 0 and avg_delay > 0 and avg_delay <= 45:
        label = 'AVERAGE_CLIENT'
    
    # 3. 1st DEGREE BAD CLIENT: less invoices + less rest + very late
    elif (invoice_count > global_avg_invoices * 0.2 and 
          invoice_count <= global_avg_invoices * 0.8 and 
          avg_remaining > global_avg_amount * 0.25 and 
          avg_remaining <= global_avg_amount * 0.75 and 
          very_late > 0):
        label = '1ST_DEGREE_BAD_CLIENT'
    
    # 2. 2nd DEGREE BAD CLIENT: TWO conditions with OR
    # Condition 1: less invoices + large rest + very late
    elif (invoice_count > global_avg_invoices * 0.2 and 
          invoice_count <= global_avg_invoices * 0.8 and 
          avg_remaining > global_avg_amount * 0.75 and 
          very_late > 0):
        label = '2ND_DEGREE_BAD_CLIENT'
    
    # Condition 2: more invoices + less rest + very late  
    elif (invoice_count > global_avg_invoices * 0.8 and 
          avg_remaining > global_avg_amount * 0.25 and 
          avg_remaining <= global_avg_amount * 0.75 and 
          very_late > 0):
        label = '2ND_DEGREE_BAD_CLIENT'
    
    # 1. 3rd DEGREE BAD CLIENT: more invoices + large rest + very late
    elif (invoice_count > global_avg_invoices * 0.8 and 
          avg_remaining > global_avg_amount * 0.75 and 
          very_late > 0):
        label = '3RD_DEGREE_BAD_CLIENT'
    
    # Default for edge cases
    else:
        if very_late > 0:
            label = '1ST_DEGREE_BAD_CLIENT'
        else:
            label = 'AVERAGE_CLIENT'
    
    labels.append(label)

clients_df['label'] = labels

# Step 4: Prepare features for ML
print("Preparing features...")
features = ['invoice_count', 'avg_amount', 'avg_remaining', 'avg_delay', 'on_time_percentage', 'very_late_payments']
X = clients_df[features]
y = clients_df['label']

# Step 5: Split data (stratified to ensure all categories in test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Train model
print("Training RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Save results
clients_df.to_csv('/home/djawed/Desktop/Ai_Training/data/classified_clients.csv', index=False)
print(f"\nResults saved to: classified_clients.csv")

# Show label distribution
print("\nLabel Distribution:")
print(clients_df['label'].value_counts())

