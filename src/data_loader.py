import pandas as pd
from sklearn.datasets import fetch_openml
import os

_cache = {}


def load_and_save_data():
    cache_key = 'porto_seguro_data'

    # Prüfe, ob der Datensatz bereits im Cache
    if cache_key in _cache:
        print("Lade Datensatz aus dem Cache.")
        return _cache[cache_key]
        print("Datensatz erfolgreich geladen")

    project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_dir = os.path.join(project_root, 'data', 'raw')
    file_path = os.path.join(data_dir, 'porto_seguro_safe_driver_prediction.csv')

    # Prüfe, ob die Datei bereits lokal vorhanden ist
    if os.path.exists(file_path):
        print("Lade Datensatz lokal...")
        df = pd.read_csv(file_path)
        _cache[cache_key] = df
        print("Datensatz erfolgreich geladen")
        return df


    # Wenn die Datei nicht vorhanden ist, lade sie von OpenML herunter
    try:
        print("Lade Datensatz von OpenML...")
        dataset = fetch_openml(data_id=42742, as_frame=True)
        df = dataset.frame

        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Datensatz erfolgreich in '{file_path}' gespeichert.")
        _cache[cache_key] = df
        return df

    except Exception as e:
        print(f"Fehler beim Laden oder Speichern: {e}")
        return None


if __name__ == "__main__":
    df_loaded = load_and_save_data()
    if df_loaded is not None:
        print(df_loaded.head())