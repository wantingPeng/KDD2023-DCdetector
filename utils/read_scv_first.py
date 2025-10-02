import pandas as pd
from pathlib import Path
from typing import Optional, Union


def read_first_n_rows(
    csv_path: Union[str, Path],
    nrows: int = 1000,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Read the first N rows of a CSV file (for quick inspection).
    
    Args:
        csv_path: Path to the CSV file.
        nrows: Number of rows to read (default: 1000).
        output_path: Optional path to save the sample (as CSV).
    
    Returns:
        DataFrame with the first N rows.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    
    # Read only the first N rows
    df = pd.read_csv(csv_path, nrows=nrows)
    
    print(df.head())


if __name__ == "__main__":
    # Example usage (modify paths as needed)
    input_csv = "dataset/ALLcontact_noSegment/test_processed.csv"
    #output_csv = "Data/data_preview/row_anomaly_data.csv"  # Optional
    

    df = read_first_n_rows(
        csv_path=input_csv,
        nrows=1000,
        #output_path=output_csv,  # Set to None if you don't want to save
    )
