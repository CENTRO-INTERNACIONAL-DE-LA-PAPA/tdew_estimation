# Test anomalies implementation

## Training

```python
import pandas as pd
from dask.distributed import Client, as_completed
from tqdm import tqdm
import os
import glob
import logging
import platform
import statsmodels.api as sm
import geopandas as gpd
import numpy as np
import shutil

# --- 1. Configuration and Constants ---

if platform.system() == "Windows":
    BASE_PATH = r"\\A019866\\Data1\\henry_simcast_peru"
else:
    BASE_PATH = "/media/ppalacios/Data1/henry_simcast_peru"

assert os.path.exists(BASE_PATH), f"Can't see data at {BASE_PATH}"

POTATO_GRID_FILE = "/media/ppalacios/Data1/henry_simcast_peru/PotatoZonning/CENAGRO_OnlyPotatoes_Pisco_Altitude.dbf"


# --- 2. Helper Functions ---

def find_parquet_files(variable: str, date_range: tuple) -> list:
    """Finds parquet files using the specified naming convention."""
    try:
        start_year, end_year = date_range
        start_date = pd.Timestamp(f"{start_year}-01-01")
        end_date = pd.Timestamp(f"{end_year}-12-31")

        months = pd.date_range(start_date, end_date, freq='MS').strftime("%Y_%m").unique()
        if variable == "tmin_v1":
            files = [f for ym in months for f in glob.glob(os.path.join(BASE_PATH, variable, "Outputs", f"tmin_daily_{ym}.parquet"))]
        else:
            files = [f for ym in months for f in glob.glob(os.path.join(BASE_PATH, variable, "Outputs", f"{variable}_daily_{ym}.parquet"))]
        return sorted(files)
    except Exception as e:
        logging.exception(f"Error in find_parquet_files for {variable}")
        return []

# --- 3. Core Logic Functions ---

def calculate_and_save_climatology_chunked(year_range, base_path, output_file):
    """Calculates daily climatology by processing data in yearly chunks to ensure memory stability."""
    print("Calculating daily climatology in yearly chunks to conserve memory...")
    start_year, end_year = year_range
    all_years = range(start_year, end_year + 1)

    temp_path = os.path.join(os.path.dirname(output_file), "climatology_temp_chunks")
    os.makedirs(temp_path, exist_ok=True)

    for year in tqdm(all_years, desc="Processing Climatology by Year"):
        chunk_file = os.path.join(temp_path, f"clim_chunk_{year}.parquet")
        if os.path.exists(chunk_file):
            continue
        date_range_chunk = (f"{year}-01-01", f"{year}-12-31")
        td_files = find_parquet_files('td', date_range_chunk)
        tmin_files = find_parquet_files('tmin_v1', date_range_chunk)
        if not td_files or not tmin_files:
            continue

        df_td_list = [pd.read_parquet(f) for f in td_files]
        df_tmin_list = [pd.read_parquet(f) for f in tmin_files]
        if not df_td_list or not df_tmin_list: continue

        df_td = pd.concat(df_td_list).rename(columns={'Value': 'TD'})
        df_tmin = pd.concat(df_tmin_list).rename(columns={'Value': 'TMIN'})

        merged = pd.merge(df_td, df_tmin, on=['FECHA', 'ID'], how='inner')
        if merged.empty: continue

        merged['doy'] = merged['FECHA'].dt.dayofyear
        agg_sum = merged.groupby(['ID', 'doy'])[['TD', 'TMIN']].sum()
        agg_count = merged.groupby(['ID', 'doy'])[['TD']].count().rename(columns={'TD': 'N'})
        yearly_stats_df = pd.merge(agg_sum, agg_count, left_index=True, right_index=True).reset_index()
        yearly_stats_df.to_parquet(chunk_file)

    print("Combining yearly statistics from disk iteratively...")
    chunk_files = sorted(glob.glob(os.path.join(temp_path, "clim_chunk_*.parquet")))
    if not chunk_files: raise FileNotFoundError("No climatology chunk files were created.")

    final_agg_df = pd.DataFrame()
    for f in tqdm(chunk_files, desc="Aggregating Chunks"):
        yearly_stats_df = pd.read_parquet(f)
        current_agg = yearly_stats_df.groupby(['ID', 'doy'])[['TD', 'TMIN', 'N']].sum()
        if final_agg_df.empty:
            final_agg_df = current_agg
        else:
            final_agg_df = final_agg_df.add(current_agg, fill_value=0)

    final_agg_df['TD_clim'] = final_agg_df['TD'] / final_agg_df['N']
    final_agg_df['TMIN_clim'] = final_agg_df['TMIN'] / final_agg_df['N']
    climatology_df = final_agg_df[['TD_clim', 'TMIN_clim']].reset_index()
    climatology_df.to_parquet(output_file, engine='pyarrow')
    print(f"✅ Climatology saved to {output_file}")
    try:
        shutil.rmtree(temp_path)
        print(f"Cleaned up temporary directory: {temp_path}")
    except Exception as e:
        print(f"Could not remove temporary directory {temp_path}: {e}")

def train_anomaly_model_for_one_id(location_id, train_year_range, h, kernel, climatology_df):
    """Loads data and fits all 366 anomaly models for ONE single ID."""
    try:
        # Filter climatology once for the relevant ID
        clim_id_df = climatology_df[climatology_df['ID'] == location_id]
        if clim_id_df.empty: return None

        td_files = find_parquet_files('td', train_year_range)
        tmin_files = find_parquet_files('tmin_v1', train_year_range)
        if not td_files or not tmin_files: return None

        df_td_list = [pd.read_parquet(f, filters=[('ID', '==', location_id)]) for f in td_files]
        df_tmin_list = [pd.read_parquet(f, filters=[('ID', '==', location_id)]) for f in tmin_files]
        df_td_list = [df for df in df_td_list if not df.empty]
        df_tmin_list = [df for df in df_tmin_list if not df.empty]
        if not df_td_list or not df_tmin_list: return None

        df_td = pd.concat(df_td_list).rename(columns={'Value': 'TD'})
        df_tmin = pd.concat(df_tmin_list).rename(columns={'Value': 'TMIN'})
        if 'TD' not in df_td.columns or 'TMIN' not in df_tmin.columns: return None

        train_df = pd.merge(df_td, df_tmin, on=['FECHA', 'ID'], how='inner')
        if train_df.empty: return None

        train_df['FECHA'] = pd.to_datetime(train_df['FECHA'])
        train_df['doy'] = train_df['FECHA'].dt.dayofyear

        # Merge with climatology to create anomalies
        train_df = pd.merge(train_df, clim_id_df, on=['ID', 'doy'], how='left')
        train_df = train_df.sort_values('FECHA').reset_index(drop=True)

        # Create anomaly and lag features
        train_df['TD_anom'] = train_df['TD'] - train_df['TD_clim']
        train_df['TMIN_anom'] = train_df['TMIN'] - train_df['TMIN_clim']
        train_df['TD_anom_lag1'] = train_df['TD_anom'].shift(1)
        train_df['TD_anom_lag2'] = train_df['TD_anom'].shift(2)
        train_df['TMIN_anom_lag1'] = train_df['TMIN_anom'].shift(1)

        results_for_id = []
        feature_cols = ['TMIN_anom', 'TD_anom_lag1', 'TD_anom_lag2', 'TMIN_anom_lag1']

        for doy_target in range(1, 367):
            lower_bound = (doy_target - h - 1) % 366 + 1
            upper_bound = (doy_target + h - 1) % 366 + 1
            mask = (train_df['doy'] >= lower_bound) & (train_df['doy'] <= upper_bound) if lower_bound < upper_bound else (train_df['doy'] >= lower_bound) | (train_df['doy'] <= upper_bound)

            df_neighborhood = train_df[mask].copy()
            df_neighborhood.dropna(subset=['TD_anom'] + feature_cols, inplace=True)

            if len(df_neighborhood) < 15: continue

            distance = np.abs(df_neighborhood['doy'] - doy_target)
            distance = np.minimum(distance, 366 - distance)
            scaled_distance = np.clip(distance / h, 0, 1)
            weights = (1 - np.abs(scaled_distance)**3)**3 if kernel == "Tricube Kernel" else np.exp(-(distance**2) / (2 * h**2))

            y = df_neighborhood['TD_anom']
            X = df_neighborhood[feature_cols]
            X = sm.add_constant(X)

            model = sm.WLS(y, X, weights=weights).fit()

            result_row = {col: model.params.get(col) for col in X.columns}
            result_row['doy'] = doy_target
            result_row['r_squared_anom'] = model.rsquared
            results_for_id.append(result_row)

        if not results_for_id: return None
        final_df = pd.DataFrame(results_for_id).rename(columns={'const': 'const_anom', 'TMIN_anom': 'TMIN_anom_coeff'})
        final_df['ID'] = location_id
        return final_df
    except Exception:
        return None

def main_anomaly():
    """Main function to parallelize the anomaly model training."""
    # ... (parameters are the same) ...
    train_year_range = (1981, 2016)
    h = 11
    kernel = "Tricube Kernel"
    results_path = "/media/ppalacios/Data1/henry_simcast_peru/results"
    climatology_file = os.path.join(results_path, "daily_climatology.parquet")
    output_file = os.path.join(results_path, "llr_coeffs_anomaly_final_direct.parquet")
    chunk_path = os.path.join(results_path, "anomaly_coeffs_chunks")
    os.makedirs(chunk_path, exist_ok=True)

    # --- PHASE 1 (Unchanged) ---
    print("--- PHASE 1: CLIMATOLOGY CALCULATION ---")
    if not os.path.exists(climatology_file):
        calculate_and_save_climatology_chunked(train_year_range, BASE_PATH, climatology_file)
    else:
        print("Climatology file already exists. Skipping calculation.")

    # --- PHASE 2 (Corrected) ---
    print("\n--- PHASE 2: ANOMALY MODEL FITTING ---")
    client = Client(n_workers=8, threads_per_worker=4, memory_limit="16GB")
    print(f"Dask Client Dashboard: {client.dashboard_link}")

    # Load climatology into pandas on the main process
    climatology_df = pd.read_parquet(climatology_file)

    # --- FIX 1: Scatter the large DataFrame to all workers once ---
    # This returns a lightweight 'future' (a pointer to the data)
    print("Scattering climatology data to all workers...")
    climatology_future = client.scatter(climatology_df, broadcast=True)

    potato_grid = gpd.read_file(POTATO_GRID_FILE)
    potato_grid["ID"] = potato_grid.get("ID", potato_grid.index)
    all_ids = potato_grid['ID'].tolist()

    id_chunk_size = 1000
    for i in range(0, len(all_ids), id_chunk_size):
        id_batch = all_ids[i : i + id_chunk_size]
        print(f"\n--- Submitting batch {i//id_chunk_size + 1} with {len(id_batch)} IDs ---")

        # --- FIX 2: Pass the lightweight future to each job ---
        futures = [client.submit(train_anomaly_model_for_one_id, id, train_year_range, h, kernel, climatology_future) for id in id_batch]

        results_list = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batch"):
            result = future.result()
            if result is not None and not result.empty:
                results_list.append(result)

        if results_list:
            batch_df = pd.concat(results_list, ignore_index=True)
            batch_output_file = os.path.join(chunk_path, f"batch_{i}.parquet")
            batch_df.to_parquet(batch_output_file, engine='pyarrow')
            print(f"✅ Batch {i//id_chunk_size + 1} completed and saved.")
        else:
            print(f"⚠️ No models were trained in batch {i//id_chunk_size + 1}.")

    # ... (Combining results logic remains the same) ...

    client.close()
    print("Dask client closed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main_anomaly()
```

## Combine anomalies

```python
import pandas as pd
import glob
import os

def combine_anomaly_chunks():
    """Finds all saved anomaly batch files and combines them."""
    print("--- Combining Anomaly Model Chunks ---")
    results_path = "/media/ppalacios/Data1/henry_simcast_peru/results"
    chunk_path = os.path.join(results_path, "anomaly_coeffs_chunks")
    output_file = os.path.join(results_path, "llr_coeffs_anomaly_final_direct.parquet")

    chunk_files = glob.glob(os.path.join(chunk_path, "batch_*.parquet"))
    if not chunk_files:
        print("❌ No batch files found to combine.")
        return

    print(f"Found {len(chunk_files)} batch files to combine.")

    # Read and concatenate all chunk files
    df_list = [pd.read_parquet(f) for f in chunk_files]
    final_df = pd.concat(df_list, ignore_index=True)

    # Save the final combined file
    final_df.to_parquet(output_file, engine='pyarrow')
    print(f"✅ Successfully combined all batches into: {output_file}")
    print(f"Final coefficient file has models for {final_df['ID'].nunique()} unique IDs.")

if __name__ == "__main__":
    combine_anomaly_chunks()
```
## generate and combine predictions

```python
import pandas as pd
from dask.distributed import Client, as_completed
from tqdm import tqdm
import os
import glob
import logging
import platform
import statsmodels.api as sm
import geopandas as gpd
import numpy as np
import shutil

# --- 1. Configuration ---
PREDICTION_YEARS = (2017, 2019)
HISTORY_END_YEAR = 2016

RESULTS_PATH = "/media/ppalacios/Data1/henry_simcast_peru/results"
BASE_PATH = "/media/ppalacios/Data1/henry_simcast_peru"
PREDICTIONS_OUTPUT_FILE = os.path.join(RESULTS_PATH, f"predictions_{PREDICTION_YEARS[0]}-{PREDICTION_YEARS[1]}.parquet")
POTATO_GRID_FILE = "/media/ppalacios/Data1/henry_simcast_peru/PotatoZonning/CENAGRO_OnlyPotatoes_Pisco_Altitude.dbf"

ANOMALY_COEFFS_FILE = os.path.join(RESULTS_PATH, "llr_coeffs_anomaly_final_direct.parquet")
CLIMATOLOGY_FILE = os.path.join(RESULTS_PATH, "daily_climatology.parquet")


# --- 2. Helper Functions ---
def find_parquet_files(variable: str, date_range: tuple) -> list:
    """Finds parquet files for a given date range of years."""
    start_year, end_year = date_range
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-12-31")
    months = pd.date_range(start_date, end_date, freq='MS').strftime("%Y_%m").unique()
    if variable == "tmin_v1":
        files = [f for ym in months for f in glob.glob(os.path.join(BASE_PATH, variable, "Outputs", f"tmin_daily_{ym}.parquet"))]
    else:
        files = [f for ym in months for f in glob.glob(os.path.join(BASE_PATH, variable, "Outputs", f"{variable}_daily_{ym}.parquet"))]
    return sorted(files)

# --- 3. Core Recursive Forecast Logic for a SINGLE ID ---
def generate_recursive_forecast_for_one_id(location_id, prediction_years, history_end_year, coeffs_filepath, clim_filepath):
    """
    Generates a multi-year forecast for a single ID by loading only the data it needs.
    """
    try:
        coeffs_df = pd.read_parquet(coeffs_filepath, filters=[('ID', '==', location_id)])
        clim_df = pd.read_parquet(clim_filepath, filters=[('ID', '==', location_id)])
        if coeffs_df.empty or clim_df.empty: return None

        required_cols = ['ID', 'FECHA', 'Value']
        hist_range = (history_end_year, history_end_year)
        hist_td_files = find_parquet_files('td', hist_range)
        hist_tmin_files = find_parquet_files('tmin_v1', hist_range)
        future_tmin_files = find_parquet_files('tmin', prediction_years) # Using corrected 'tmin' path

        hist_td = pd.concat([pd.read_parquet(f, columns=required_cols, filters=[('ID', '==', location_id)]) for f in hist_td_files]).rename(columns={'Value': 'TD'})
        hist_tmin = pd.concat([pd.read_parquet(f, columns=required_cols, filters=[('ID', '==', location_id)]) for f in hist_tmin_files]).rename(columns={'Value': 'TMIN'})
        future_tmin = pd.concat([pd.read_parquet(f, columns=required_cols, filters=[('ID', '==', location_id)]) for f in future_tmin_files]).rename(columns={'Value': 'TMIN'})

        if hist_td.empty or hist_tmin.empty or future_tmin.empty: return None

        history_df = pd.merge(hist_td, hist_tmin, on=['ID', 'FECHA']).sort_values('FECHA')
        if len(history_df) < 2: return None

        last_two_days = history_df.tail(2).copy()

        clim_id_df = clim_df.set_index('doy')
        coeffs_id_df = coeffs_df.set_index('doy')
        future_tmin = future_tmin.set_index('FECHA')

        prediction_dates = pd.date_range(start=f"{prediction_years[0]}-01-01", end=f"{prediction_years[1]}-12-31")
        predictions = []

        for current_day in prediction_dates:
            doy = current_day.dayofyear

            if doy not in clim_id_df.index or doy not in coeffs_id_df.index or current_day not in future_tmin.index:
                continue

            tmin_today = future_tmin.loc[current_day, 'TMIN']
            td_lag1 = last_two_days.iloc[-1]['TD']
            td_lag2 = last_two_days.iloc[-2]['TD']
            tmin_lag1 = last_two_days.iloc[-1]['TMIN']

            clim_today = clim_id_df.loc[doy]
            clim_lag1 = clim_id_df.loc[(current_day - pd.Timedelta(days=1)).dayofyear]
            clim_lag2 = clim_id_df.loc[(current_day - pd.Timedelta(days=2)).dayofyear]

            tmin_anom = tmin_today - clim_today['TMIN_clim']
            td_anom_lag1 = td_lag1 - clim_lag1['TD_clim']
            td_anom_lag2 = td_lag2 - clim_lag2['TD_clim']
            tmin_anom_lag1 = tmin_lag1 - clim_lag1['TMIN_clim']

            coeffs = coeffs_id_df.loc[doy]

            # --- THE FIX ---
            # Use the correct column names from your coefficient file
            predicted_anomaly = (
                coeffs['const_anom'] +
                (tmin_anom * coeffs['TMIN_anom_coeff']) +
                (td_anom_lag1 * coeffs['TD_anom_lag1']) +
                (td_anom_lag2 * coeffs['TD_anom_lag2']) +
                (tmin_anom_lag1 * coeffs['TMIN_anom_lag1'])
            )

            predicted_td = predicted_anomaly + clim_today['TD_clim']
            predictions.append({'FECHA': current_day, 'ID': location_id, 'TD_predicted': predicted_td})

            new_row = pd.DataFrame([{'FECHA': current_day, 'ID': location_id, 'TD': predicted_td, 'TMIN': tmin_today}])
            last_two_days = pd.concat([last_two_days.iloc[1:], new_row], ignore_index=True)

        return pd.DataFrame(predictions)

    except Exception:
        # Fail gracefully on the worker and return None
        return None

# --- 4. Main Execution and Post-processing ---
def main():
    """Main function to parallelize the recursive forecast."""
    results_path = "/media/ppalacios/Data1/henry_simcast_peru/results"
    chunk_path = os.path.join(results_path, "prediction_chunks")
    os.makedirs(chunk_path, exist_ok=True)

    client = Client(n_workers=8, threads_per_worker=4, memory_limit="16GB")
    print(f"Dask Client Dashboard: {client.dashboard_link}")

    potato_grid = gpd.read_file(POTATO_GRID_FILE)
    potato_grid["ID"] = potato_grid.get("ID", potato_grid.index)
    all_ids = potato_grid['ID'].tolist()

    id_chunk_size = 500
    for i in range(0, len(all_ids), id_chunk_size):
        id_batch = all_ids[i : i + id_chunk_size]
        print(f"\n--- Submitting batch {i//id_chunk_size + 1} with {len(id_batch)} IDs ---")

        futures = [
            client.submit(
                generate_recursive_forecast_for_one_id,
                id,
                PREDICTION_YEARS,
                HISTORY_END_YEAR,
                ANOMALY_COEFFS_FILE,
                CLIMATOLOGY_FILE
            ) for id in id_batch
        ]

        results_list = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batch"):
            result = future.result()
            if result is not None and not result.empty:
                results_list.append(result)

        if results_list:
            batch_df = pd.concat(results_list, ignore_index=True)
            batch_output_file = os.path.join(chunk_path, f"pred_batch_{i}.parquet")
            batch_df.to_parquet(batch_output_file, engine='pyarrow')
            print(f"✅ Batch {i//id_chunk_size + 1} completed and saved.")
        else:
            print(f"⚠️ No predictions were generated in batch {i//id_chunk_size + 1}.")

    print("\nAll batches processed.")
    client.close()
    print("Dask client closed.")

def combine_prediction_chunks():
    """Finds all saved prediction chunk files and combines them."""
    print("\n--- Combining all prediction chunks ---")
    results_path = "/media/ppalacios/Data1/henry_simcast_peru/results"
    chunk_path = os.path.join(results_path, "prediction_chunks")

    chunk_files = glob.glob(os.path.join(chunk_path, "pred_batch_*.parquet"))
    if not chunk_files:
        print("❌ No prediction chunk files were found to combine.")
        return

    print(f"Found {len(chunk_files)} chunk files to combine.")
    final_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)

    final_df.to_parquet(PREDICTIONS_OUTPUT_FILE, engine='pyarrow')
    print(f"✅ Successfully saved final predictions for {final_df['ID'].nunique()} IDs to {PREDICTIONS_OUTPUT_FILE}")

    try:
        shutil.rmtree(chunk_path)
        print(f"Cleaned up temporary directory: {chunk_path}")
    except Exception as e:
        print(f"Could not remove temporary directory {chunk_path}: {e}")

if __name__ == "__main__":
    main()
    combine_prediction_chunks()
```

# split prediciotns into different files per month

```python
import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
import os
from tqdm import tqdm

# --- 1. Configuration ---

# The large prediction file you want to split
INPUT_FILE = "/media/ppalacios/Data1/henry_simcast_peru/results/predictions_2017-2019.parquet"

# The final destination for the new monthly files
OUTPUT_PATH = "/media/ppalacios/Data1/henry/simcast_peru/td_predicted/Outputs"


def split_predictions_dask():
    """
    Loads a single large prediction file and splits it into
    multiple monthly parquet files using Dask.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found at {INPUT_FILE}")
        return

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    client = Client(n_workers=8, threads_per_worker=4, memory_limit="16GB")
    print(f"Dask Client Dashboard: {client.dashboard_link}")

    # 1. Load the large file lazily into a Dask DataFrame
    print(f"Reading file: {INPUT_FILE}")
    predictions_ddf = dd.read_parquet(INPUT_FILE)

    # 2. Create the 'year_month' column for grouping
    predictions_ddf['FECHA'] = dd.to_datetime(predictions_ddf['FECHA'])
    predictions_ddf['year_month'] = predictions_ddf['FECHA'].dt.strftime('%Y_%m')

    # 3. Get the unique list of months to process
    print("Finding unique months in the dataset...")
    unique_months = predictions_ddf['year_month'].unique().compute()
    print(f"Found {len(unique_months)} months to process.")

    # 4. Loop through each month and save its data
    for year_month in tqdm(sorted(unique_months), desc="Saving Monthly Files"):

        # Filter the Dask DataFrame to get only the data for the current month
        month_ddf = predictions_ddf[predictions_ddf['year_month'] == year_month]

        # Prepare the output DataFrame: select and rename columns
        output_ddf = month_ddf[['ID', 'FECHA', 'TD_predicted']].rename(columns={'TD_predicted': 'Value'})

        # Bring the result for this single month into a pandas DataFrame
        # This is safe as one month of data is small enough to fit in memory
        output_pandas_df = output_ddf.compute()

        # Construct the final filename
        output_filename = os.path.join(OUTPUT_PATH, f"td_daily_{year_month}.parquet")

        # Save the pandas DataFrame as a single Parquet file
        output_pandas_df.to_parquet(output_filename, engine='pyarrow', index=False)

    print("\n" + "="*50)
    print("✅ Splitting complete!")
    print(f"All monthly prediction files have been saved to:\n{OUTPUT_PATH}")
    print("="*50)

    client.close()
    print("Dask client closed.")


if __name__ == "__main__":
    split_predictions_dask()
```

## Comparative of non anomaly vs anomaly

```python
import pandas as pd
import numpy as np
import os
import glob
import logging
from sklearn.metrics import mean_squared_error, r2_score

# --- Configuration and File Paths ---
RESULTS_PATH = "/media/ppalacios/Data1/henry_simcast_peru/results"
BASE_PATH = "/media/ppalacios/Data1/henry_simcast_peru"
# Use a year that was NOT in your training data (e.g., 1981-2016)
TEST_YEAR = 2004

RAW_COEFFS_FILE = os.path.join(RESULTS_PATH, "llr_coeffs_raw_final.parquet")
ANOMALY_COEFFS_FILE = os.path.join(RESULTS_PATH, "llr_coeffs_anomaly_final.parquet")
CLIMATOLOGY_FILE = os.path.join(RESULTS_PATH, "daily_climatology.parquet")


def find_parquet_files(variable: str, date_range: tuple) -> list:
    """
    Finds parquet files for a given date range. This version is robust.
    """
    # --- CORRECTED LOGIC ---
    start, end = date_range
    if isinstance(start, int) and isinstance(end, int):
        # Handles tuples of years like (1981, 2016)
        start_date = pd.Timestamp(f"{start}-01-01")
        end_date = pd.Timestamp(f"{end}-12-31")
    else:
        # Handles tuples of timestamps or date strings
        start_date, end_date = [pd.Timestamp(d) for d in date_range]

    months = pd.date_range(start_date, end_date, freq='MS').strftime("%Y_%m").unique()
    if variable == "tmin_v1":
        files = [f for ym in months for f in glob.glob(os.path.join(BASE_PATH, variable, "Outputs", f"tmin_daily_{ym}.parquet"))]
    else:
        files = [f for ym in months for f in glob.glob(os.path.join(BASE_PATH, variable, "Outputs", f"{variable}_daily_{ym}.parquet"))]
    return sorted(files)

def prepare_test_data(year: int, clim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads and prepares all necessary features (raw and anomaly) for a given test year.
    """
    print(f"Loading and preparing all test data for {year}...")
    date_range = (f"{year}-01-01", f"{year}-12-31")

    td_files = find_parquet_files('td', date_range)
    tmin_files = find_parquet_files('tmin_v1', date_range)

    if not td_files or not tmin_files:
        raise FileNotFoundError(f"No test data found for the year {year}.")

    df_td = pd.concat([pd.read_parquet(f) for f in td_files], ignore_index=True)
    df_tmin = pd.concat([pd.read_parquet(f) for f in tmin_files], ignore_index=True)

    df_td = df_td.rename(columns={'Value': 'TD'})
    df_tmin = df_tmin.rename(columns={'Value': 'TMIN'})

    test_df = pd.merge(df_td, df_tmin, on=['ID', 'FECHA'], how='inner')
    test_df['doy'] = pd.to_datetime(test_df['FECHA']).dt.dayofyear

    test_df = test_df.sort_values(['ID', 'FECHA'])

    # Raw lags
    test_df['TD_lag1'] = test_df.groupby('ID')['TD'].shift(1)
    test_df['TD_lag2'] = test_df.groupby('ID')['TD'].shift(2)
    test_df['TMIN_lag1'] = test_df.groupby('ID')['TMIN'].shift(1)

    # Merge climatology
    test_df = pd.merge(test_df, clim_df, on=['ID', 'doy'], how='left')

    # Anomalies
    test_df['TMIN_anom'] = test_df['TMIN'] - test_df['TMIN_clim']
    test_df['TD_anom'] = test_df['TD'] - test_df['TD_clim']

    # Anomaly lags
    test_df['TD_anom_lag1'] = test_df.groupby('ID')['TD_anom'].shift(1)
    test_df['TD_anom_lag2'] = test_df.groupby('ID')['TD_anom'].shift(2)
    test_df['TMIN_anom_lag1'] = test_df.groupby('ID')['TMIN_anom'].shift(1)

    return test_df.dropna().reset_index(drop=True)

def predict_non_anomaly(test_df: pd.DataFrame, coeffs_df: pd.DataFrame) -> pd.DataFrame:
    """Makes predictions using the non-anomaly model on pre-prepared data."""
    print("Making predictions with non-anomaly model...")
    # Add suffixes to avoid column name collisions
    merged_df = pd.merge(test_df, coeffs_df, on=['ID', 'doy'], suffixes=('_val', '_coeff'))

    merged_df['prediction'] = (
        merged_df['const'] +
        (merged_df['TMIN_val'] * merged_df['TMIN_coeff']) +
        (merged_df['TD_lag1_val'] * merged_df['TD_lag1_coeff']) +
        (merged_df['TD_lag2_val'] * merged_df['TD_lag2_coeff']) +
        (merged_df['TMIN_lag1_val'] * merged_df['TMIN_lag1_coeff'])
    )
    return merged_df

def predict_anomaly(test_df: pd.DataFrame, coeffs_df: pd.DataFrame) -> pd.DataFrame:
    """Makes predictions using the anomaly model on pre-prepared data."""
    print("Making predictions with anomaly model...")
    merged_df = pd.merge(test_df, coeffs_df, on=['ID', 'doy'], suffixes=('_val', '_coeff'))

    predicted_anomaly = (
        merged_df['const'] +
        (merged_df['TMIN_anom_val'] * merged_df['TMIN_anom_coeff']) +
        (merged_df['TD_anom_lag1_val'] * merged_df['TD_anom_lag1_coeff']) +
        (merged_df['TD_anom_lag2_val'] * merged_df['TD_anom_lag2_coeff']) +
        (merged_df['TMIN_anom_lag1_val'] * merged_df['TMIN_anom_lag1_coeff'])
    )

    merged_df['prediction'] = predicted_anomaly + merged_df['TD_clim']

    return merged_df


if __name__ == "__main__":
    print("Loading coefficients and climatology files...")
    raw_coeffs = pd.read_parquet(RAW_COEFFS_FILE)
    anomaly_coeffs = pd.read_parquet(ANOMALY_COEFFS_FILE)
    climatology = pd.read_parquet(CLIMATOLOGY_FILE)

    # Prepare all test data features in one go
    test_data = prepare_test_data(TEST_YEAR, climatology)

    # Generate predictions using the simplified functions
    results_raw = predict_non_anomaly(test_data, raw_coeffs)
    results_anomaly = predict_anomaly(test_data, anomaly_coeffs)

    # Find common rows to ensure a fair comparison
    common_indices = results_raw.index.intersection(results_anomaly.index)

    results_raw_common = results_raw.loc[common_indices]
    results_anomaly_common = results_anomaly.loc[common_indices]

    # Calculate final metrics on the common data points
    rmse_raw = np.sqrt(mean_squared_error(results_raw_common['TD'], results_raw_common['prediction']))
    rmse_anomaly = np.sqrt(mean_squared_error(results_anomaly_common['TD'], results_anomaly_common['prediction']))

    r2_raw = r2_score(results_raw_common['TD'], results_raw_common['prediction'])
    r2_anomaly = r2_score(results_anomaly_common['TD'], results_anomaly_common['prediction'])

    print("\n" + "="*50)
    print("       MODEL PERFORMANCE COMPARISON")
    print("="*50)
    print(f"Test Year: {TEST_YEAR}")
    print(f"Data Points Compared: {len(common_indices)}")
    print("-"*50)
    print(f"Non-Anomaly Model -> RMSE: {rmse_raw:.4f} | R-squared: {r2_raw:.4f}")
    print(f"Anomaly Model     -> RMSE: {rmse_anomaly:.4f} | R-squared: {r2_anomaly:.4f}")
    print("="*50)
    if rmse_anomaly < rmse_raw:
        print("🏆 The Anomaly Model is better (Lower RMSE).")
    else:
        print("🏆 The Non-Anomaly Model is better (Lower RMSE).")
    print("="*50)
```
