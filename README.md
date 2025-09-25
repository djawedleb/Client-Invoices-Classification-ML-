# AI Training - Client Classification System

A machine learning project that classifies clients based on their payment behavior and invoice patterns using invoice data.

## Overview

This project analyzes client payment patterns from invoice data to classify clients into different categories based on their payment reliability, amounts, and timing. The system uses a Random Forest classifier to predict client categories.

## Project Structure

```
Ai_Training/
├── data/                          # Data files
│   ├── invoices.csv              # Training invoice data
│   ├── invoice2.csv              # Additional invoice data
│   └── training_results.csv      # Model training results
├── model/                         # Trained models
│   └── client_classifier_model.pkl # Trained Random Forest model
├── notebooks/                     # Jupyter notebooks
│   └── 01_data_exploration.ipynb # Data exploration and analysis
├── scripts/                       # Python scripts
│   ├── simple_classifier.py      # Model training script
│   ├── predict_clients.py        # Prediction script
│   └── test.py                   # Testing utilities
└── requirements.txt              # Python dependencies
```

## Features

The system analyzes the following client features:
- **Invoice Count**: Number of invoices per client
- **Average Amount**: Average invoice amount per client
- **Average Remaining**: Average outstanding amount per client
- **Average Delay**: Average payment delay in days
- **On-time Percentage**: Percentage of payments made on time
- **Very Late Payments**: Count of payments >45 days late

## Client Categories

The model classifies clients into the following categories:
- **EXCELLENT_CLIENT**: 100% on-time payments
- **GOOD_CLIENT**: 80-99% on-time payments with low average delay
- **AVERAGE_CLIENT**: Some delay but not very late
- **1ST_DEGREE_BAD_CLIENT**: Has very late payments with low remaining debt
- **2ND_DEGREE_BAD_CLIENT**: Has very late payments with medium remaining debt
- **3RD_DEGREE_BAD_CLIENT**: Has very late payments with high remaining debt

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train a new model with your invoice data:

```bash
python scripts/simple_classifier.py
```

This will:
- Load invoice data from `data/invoices.csv`
- Process and create client features
- Train a Random Forest classifier
- Save the model to `model/client_classifier_model.pkl`
- Save training results to `data/training_results.csv`

### Making Predictions

To classify clients from new invoice data:

```bash
python scripts/predict_clients.py <input_csv> <output_csv> [model_path]
```

Example:
```bash
python scripts/predict_clients.py data/invoices.csv data/predictions.csv
```

### Data Exploration

Open the Jupyter notebook to explore the data:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Data Format

The input CSV should contain the following columns:
- `id_facture`: Invoice ID
- `nom_du_client`: Client name
- `create_date`: Invoice creation date
- `due_in_date`: Payment due date
- `clear_date`: Payment completion date
- `montant_totale`: Total invoice amount
- `totale_payer`: Amount paid
- `totale_reste`: Remaining amount

## Dependencies

- scikit-learn: Machine learning library
- pandas: Data manipulation
- numpy: Numerical computing
- matplotlib & seaborn: Data visualization
- joblib: Model persistence
- jupyter: Notebook environment

## Model Performance

The model uses a Random Forest classifier with:
- 200 estimators
- Balanced class weights
- 80/20 train-test split
- Stratified sampling

## Output

The prediction script generates a CSV file with:
- Client names
- Calculated features
- Predicted client categories
- Category distribution summary

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## License

This project is for educational and research purposes.
