# tools/format_dataset.py

import os
import pandas as pd

def load_meld_split(split_csv_path):
    """
    Load MELD dataset split CSV and return a cleaned DataFrame.
    """
    df = pd.read_csv(split_csv_path)
    # Select relevant columns like 'Utterance', 'Emotion' (adjust as per your MELD CSV)
    df = df[['Utterance', 'Emotion']]
    # Filter out samples with unknown or ambiguous emotions if needed
    df = df[df['Emotion'] != 'non-neutral']  # example if you want to filter
    df = df.dropna()
    return df

def save_formatted_dataset(df, out_path):
    """
    Save cleaned and formatted dataset to CSV.
    """
    df.to_csv(out_path, index=False)
    print(f"Saved formatted dataset to {out_path}")

def main():
    # Change these paths as per your data folder structure
    meld_data_dir = "./data/datasets/MELD"
    splits = ['train_sent_emo.csv', 'dev_sent_emo.csv', 'test_sent_emo.csv']

    for split in splits:
        input_path = os.path.join(meld_data_dir, split)
        df = load_meld_split(input_path)

        out_path = os.path.join(meld_data_dir, "formatted_" + split)
        save_formatted_dataset(df, out_path)

if __name__ == "__main__":
    main()
