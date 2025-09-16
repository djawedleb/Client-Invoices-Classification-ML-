import sys

import joblib
import pandas as pd


DEFAULT_MODEL_PATH = 'model/client_classifier_model.pkl'
 
#  run command python3 /home/djawed/Desktop/Ai_Training/scripts/predict_clients.py /home/djawed/Desktop/Ai_Training/data/invoices_v2.csv /home/djawed/Desktop/Ai_Training/data/predictions_other.csv

def main():
    if len(sys.argv) < 3:
        print('Usage: python3 scripts/predict_clients.py <input_csv> <output_csv> [model_path]')
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL_PATH

    # Load model
    model = joblib.load(model_path)

    # Read invoices CSV (no training data needed)
    df = pd.read_csv(input_csv)

    # Ensure datetime types
    df['create_date'] = pd.to_datetime(df['create_date'])
    df['due_in_date'] = pd.to_datetime(df['due_in_date'])
    df['clear_date'] = pd.to_datetime(df['clear_date'])

    # Payment delay in days
    df['payment_delay'] = (df['clear_date'] - df['due_in_date']).dt.days

    # Aggregate features per client (same as training)
    rows = []
    for client in df['nom_du_client'].unique():
        cdf = df[df['nom_du_client'] == client]
        invoice_count = len(cdf)
        avg_amount = cdf['montant_totale'].mean()
        avg_remaining = cdf['totale_reste'].mean()
        avg_delay = cdf['payment_delay'].mean()
        on_time = len(cdf[cdf['payment_delay'] <= 0])
        total = len(cdf[cdf['payment_delay'].notna()])
        on_time_percentage = (on_time / total * 100) if total > 0 else 0
        very_late_payments = len(cdf[cdf['payment_delay'] > 45])
        rows.append({
            'client_name': client,
            'invoice_count': invoice_count,
            'avg_amount': avg_amount,
            'avg_remaining': avg_remaining,
            'avg_delay': avg_delay,
            'on_time_percentage': on_time_percentage,
            'very_late_payments': very_late_payments
        })

    clients_df = pd.DataFrame(rows)

    # Prepare features (must match training)
    features = ['invoice_count', 'avg_amount', 'avg_remaining', 'avg_delay', 'on_time_percentage', 'very_late_payments']
    X = clients_df[features].fillna(0)

    # Predict
    preds = model.predict(X)
    clients_df['predicted_label'] = preds

    # Print category counts and total
    print("Predicted categories count:")
    counts = clients_df['predicted_label'].value_counts()
    for label, cnt in counts.items():
        print(f"- {label}: {cnt}")
    print(f"Total: {len(clients_df)}")

    # Save
    clients_df.to_csv(output_csv, index=False)
    print(f'Saved predictions to {output_csv}')


if __name__ == '__main__':
    main()


