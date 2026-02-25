# ingest.py
import pandas as pd
import utils as e_util

def main():
    
    csv_path = 'synthetic_transactions.csv'
    
    df = e_util.generate_synthetic_transactions(
    num_transactions=10000,
    years=2,
    output_file='synthetic_transactions.csv',
    combine=True,
    annual_revenue_min=500000,
    annual_revenue_max=1300000
    )
    
    try:
        e_util.ingest_data(csv_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Check that CORPUS_FILE_PATH in your .env points to a valid file.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()