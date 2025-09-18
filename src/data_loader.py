import pandas as pd
from sklearn.datasets import fetch_openml
import os
from pathlib import Path

_cache = {}


def load_and_save_data():
    """
    Loads the Porto Seguro dataset from a local cache if available,
    otherwise fetches it from OpenML and saves it locally.
    """
    cache_key = 'porto_seguro_data'

    # Check if the dataset is already in the in-memory cache
    if cache_key in _cache:
        print("Loading dataset from in-memory cache.")
        return _cache[cache_key]

    # Determine project root and file path
    # This assumes the script is in a standard project structure (e.g., in 'src' or root)
    try:
        project_root = Path(__file__).resolve().parents[1]
    except NameError:
        project_root = Path.cwd().parent if Path.cwd().name in ["notebooks", "src"] else Path.cwd()

    data_dir = project_root / 'data' / 'raw'
    file_path = data_dir / 'porto_seguro_safe_driver_prediction.csv'

    # Check if the file already exists locally
    if file_path.exists():
        print(f"Loading dataset from local file: {file_path}")
        df = pd.read_csv(file_path)
        _cache[cache_key] = df
        print("Dataset loaded successfully.")
        return df

    # If the file doesn't exist, download it from OpenML
    try:
        print("Fetching dataset from OpenML...")
        dataset = fetch_openml(data_id=42742, as_frame=True, parser='auto')
        df = dataset.frame

        # Create directory if it doesn't exist and save the file
        data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Dataset successfully saved to '{file_path}'.")
        _cache[cache_key] = df
        return df

    except Exception as e:
        print(f"Error during download or saving: {e}")
        return None


if __name__ == "__main__":
    df_loaded = load_and_save_data()
    if df_loaded is not None:
        print(df_loaded.head())