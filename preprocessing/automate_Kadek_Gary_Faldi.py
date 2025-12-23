import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import sys

def run_preprocessing(input_path, output_path):
    df = pd.read_csv(input_path)
    print(f"Memproses: {input_path}...")

    cols_to_drop = ['Unnamed: 0', 'id']
    df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)

    if 'Arrival Delay in Minutes' in df.columns:
        median_val = df['Arrival Delay in Minutes'].median()
        df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(median_val)

    le = LabelEncoder()
    cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()
    num_cols = [
        'Age', 'Flight Distance', 'Inflight wifi service',
        'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
        'Inflight entertainment', 'On-board service', 'Leg room service',
        'Baggage handling', 'Checkin service', 'Inflight service',
        'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'
    ]

    existing_num_cols = [c for c in num_cols if c in df.columns]
    df[existing_num_cols] = scaler.fit_transform(df[existing_num_cols])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Output disimpan di: {output_path}")


if __name__ == "__main__":
    files = ["train.csv", "test.csv"]
    base_input_dir = "Airline Passenger Satisfaction"
    base_output_dir = "preprocessing/Airline Passenger Satisfaction_Cleaned"

    missing_files = []

    for file_name in files:
        input_path = os.path.join(base_input_dir, file_name)
        if not os.path.exists(input_path):
            missing_files.append(input_path)

    if missing_files:
        print("ERROR: File input berikut tidak ditemukan:")
        for f in missing_files:
            print(f"- {f}")
        print("Preprocessing dihentikan.")
        sys.exit(1)

    for file_name in files:
        raw_path = os.path.join(base_input_dir, file_name)
        processed_path = os.path.join(
            base_output_dir,
            file_name.replace(".csv", "_cleaned.csv")
        )
        run_preprocessing(raw_path, processed_path)
