from pathlib import Path
import pandas as pd

RAW = Path("data/raw/dft_road_safety_last_5_years")
PROC = Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)

# Detect naming (some bundles use Collisions.csv instead of Accidents.csv)
ACC_FILE = RAW / ("Accidents.csv" if (RAW / "Accidents.csv").exists() else "Collisions.csv")
VEH_FILE = RAW / "Vehicles.csv"
CAS_FILE = RAW / "Casualties.csv"

def read_csv_fast(path: Path) -> pd.DataFrame:
    # PyArrow engine is fast + lower mem than default
    return pd.read_csv(path, engine="pyarrow")

def normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})
    return df

def main():
    # --- load
    acc = normalise_cols(read_csv_fast(ACC_FILE))
    veh = normalise_cols(read_csv_fast(VEH_FILE))
    cas = normalise_cols(read_csv_fast(CAS_FILE))

    # --- key columns (handle variants across years)
    # accidents/collisions table
    acc_id = "accident_index"
    # vehicles + casualties tables always reference accident_index
    veh_id, veh_ref = "accident_index", "vehicle_reference"
    cas_id, cas_ref = "accident_index", "vehicle_reference"

    # --- minimal type fixes (dates)
    # date or accident_date depending on release
    date_col = next((c for c in acc.columns if c in ["date","accident_date"]), None)
    if date_col:
        acc[date_col] = pd.to_datetime(acc[date_col], errors="coerce", dayfirst=True)
        acc["year"]  = acc[date_col].dt.year
        acc["month"] = acc[date_col].dt.to_period("M").dt.to_timestamp()

    # --- join vehicles -> casualties (on accident + vehicle)
    cas = cas.merge(veh[[veh_id, veh_ref]], how="left", left_on=[cas_id, cas_ref], right_on=[veh_id, veh_ref], suffixes=("",""))

    # --- join accidents -> casualties (1-to-many)
    joined = cas.merge(
        acc,
        how="left",
        left_on=cas_id,
        right_on=acc_id,
        suffixes=("_cas","_acc")
    )

    # Keep a tidy subset for EDA / modeling (add more cols later)
    keep_cols = [
        # IDs
        cas_id, cas_ref,
        # target / outcomes
        "severity" if "severity" in joined.columns else "casualty_severity",
        # casualty dims
        "casualty_class","sex_of_casualty","age_band_of_casualty",
        # vehicle dims (via link)
        "sex_of_driver" if "sex_of_driver" in joined.columns else None,
        "age_band_of_driver" if "age_band_of_driver" in joined.columns else None,
        "vehicle_type" if "vehicle_type" in joined.columns else None,
        # accident dims
        date_col, "year","month",
        "light_conditions","weather_conditions","road_type","speed_limit",
        "local_authority_(district)" if "local_authority_(district)" in joined.columns else None,
        "police_force" if "police_force" in joined.columns else None,
        "number_of_vehicles" if "number_of_vehicles" in joined.columns else None,
        "number_of_casualties" if "number_of_casualties" in joined.columns else None,
        "longitude" if "longitude" in joined.columns else None,
        "latitude"  if "latitude"  in joined.columns else None,
    ]
    keep_cols = [c for c in keep_cols if c and c in joined.columns]
    tidy = joined[keep_cols].copy()

    # Standardise target column name
    if "casualty_severity" in tidy.columns and "severity" not in tidy.columns:
        tidy = tidy.rename(columns={"casualty_severity":"severity"})

    # Save fast format
    out = PROC / "casualty_joined.parquet"
    tidy.to_parquet(out, index=False)
    print(f"Saved {out} with {len(tidy):,} rows and {len(tidy.columns)} columns")

if __name__ == "__main__":
    main()
