import pandas as pd
from sklearn.datasets import fetch_openml
import os

# Globale Variable, um den DataFrame zwischenzuspeichern
_df = None


def load_and_save_data(cache=True):
    global _df

    data_dir = 'data/raw/'
    file_path = os.path.join(data_dir, 'porto_seguro_safe_driver_prediction.csv')

    if _df is not None and cache:
        print("Lade Datensatz aus dem Cache.")
        return _df

    # Prüfen, ob die Datei bereits existiert
    if os.path.exists(file_path):
        print("Lade Datensatz von lokaler Datei...")
        df = pd.read_csv(file_path)
        if cache:
            _df = df
        return df

    try:
        print("Lade Datensatz von OpenML (ID: 42742)...")
        dataset = fetch_openml(data_id=42742, as_frame=True)
        df = dataset.frame

        # Sicherstellen, dass das Verzeichnis existiert
        os.makedirs(data_dir, exist_ok=True)

        # Speichern des Datensatzes
        df.to_csv(file_path, index=False)
        print(f"Datensatz erfolgreich in '{file_path}' gespeichert.")

        if cache:
            _df = df

        return df

    except Exception as e:
        print(f"Fehler beim Laden oder Speichern des Datensatzes: {e}")
        return None


if __name__ == "__main__":
    # Beispielnutzung, wenn die Datei direkt ausgeführt wird
    df_loaded = load_and_save_data()
    if df_loaded is not None:
        print("\nErste 5 Zeilen des geladenen Datensatzes:")
        print(df_loaded.head())