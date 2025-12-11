import pandas as pd
import sqlite3
import os
from pathlib import Path

class IndustrialETL:
    """
    Handles the Extract, Transform, Load (ETL) pipeline for industrial sensor data.
    Simulates moving data from a raw CSV export into a production SQL database.
    """
    
    def __init__(self, db_path: str, raw_data_path: str):
        self.db_path = db_path
        self.raw_data_path = raw_data_path
        self.conn = None

    def _get_connection(self):
        """Creates a connection to the SQLite database."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        return sqlite3.connect(self.db_path)

    def extract(self) -> pd.DataFrame:
        """EXTRACT: Reads the raw CSV file."""
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"❌ Raw data not found at: {self.raw_data_path}")
        
        print(f"📥 Extracting data from {self.raw_data_path}...")
        df = pd.read_csv(self.raw_data_path)
        print(f"   Rows extracted: {len(df)}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """TRANSFORM: Cleans data and formats types for SQL storage."""
        print("⚙️ Transforming data...")
        
        # 1. Remove CSV artifacts
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
            
        # 2. Fix Timestamps (Critical for Time-Series)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 3. Drop 'sensor_15' (Business Logic: Known broken sensor)
        if 'sensor_15' in df.columns:
            df.drop(columns=['sensor_15'], inplace=True)
            print("   Dropped known broken sensor: sensor_15")

        # 4. Handle Categorical Target for SQL
        # Ensure machine_status is clean string
        df['machine_status'] = df['machine_status'].astype(str)
        
        print(f"   Transformed shape: {df.shape}")
        return df

    def load(self, df: pd.DataFrame, table_name: str = "sensor_logs"):
        """LOAD: Writes the clean data into SQLite."""
        print(f"💾 Loading data into SQLite DB ({self.db_path})...")
        
        try:
            self.conn = self._get_connection()
            # if_exists='replace' refreshes the DB. Use 'append' for real streaming.
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)
            print(f"✅ Success! Data loaded into table '{table_name}'.")
            
            # Verify insertion
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT count(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"   Total rows in DB: {count}")
            
        except Exception as e:
            print(f"❌ Database Error: {e}")
        finally:
            if self.conn:
                self.conn.close()

    def run_pipeline(self):
        """Executes the full ETL process."""
        df_raw = self.extract()
        df_clean = self.transform(df_raw)
        self.load(df_clean)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Define paths relative to this script
    # Assuming script is in /src/ and data is in /data/
    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_CSV = BASE_DIR / "data" / "raw" / "sensor.csv"
    DB_FILE = BASE_DIR / "data" / "database.db"
    
    etl = IndustrialETL(db_path=str(DB_FILE), raw_data_path=str(RAW_CSV))
    etl.run_pipeline()