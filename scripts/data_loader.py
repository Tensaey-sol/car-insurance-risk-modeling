import pandas as pd
import csv

def convert_txt_to_csv(input_path: str, output_path: str, delimiter: str = '|'):
    """
    Convert pipe-delimited text file to CSV.
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile, delimiter=delimiter)
        writer = csv.writer(outfile)
        for row in reader:
            writer.writerow(row)

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV file into pandas DataFrame.
    """
    return pd.read_csv(csv_path)