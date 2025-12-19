import logging
import os
from datetime import datetime

import pandas as pd
import src.data_contract as dc

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Defensive loader:
    - Reads parquet
    - Validates required schema
    - Sanitizes rows using contract rules
    - Writes a sample of dropped rows to a writable artifact dir (NOT project root)
    - Attaches stats + artifact paths on df.attrs for main.py to log
    """

    def __init__(self, path: str, artifact_dir: str | None = None):
        self.path = path
        self.artifact_dir = artifact_dir or os.environ.get("LOCAL_ARTIFACT_DIR", "/tmp/mlops_artifacts")

    def _ensure_artifact_dir(self) -> str:
        os.makedirs(self.artifact_dir, exist_ok=True)
        return self.artifact_dir

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.path}...")

        try:
            df = pd.read_parquet(self.path)
        except Exception as e:
            logger.error(f"Failed to read parquet file: {e}")
            raise

        # 1) Critical schema check (fail fast)
        missing_cols = [c for c in dc.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Schema Violation: Missing columns {missing_cols}")

        raw_count = len(df)
        logger.info(f"Initial row count: {raw_count}")

        # 2) Type enforcement
        if "payment_type" in df.columns:
            df["payment_type"] = pd.to_numeric(df["payment_type"], errors="coerce")

        # 3) Drop nulls in required columns
        df = df.dropna(subset=dc.REQUIRED_COLUMNS)
        after_null_drop = len(df)

        # 4) Calculate rule masks (observability)
        mask_passenger = (df["passenger_count"] >= dc.PASSENGER_MIN) & (df["passenger_count"] <= dc.PASSENGER_MAX)
        mask_distance = (df["trip_distance"] >= dc.TRIP_DISTANCE_MIN) & (df["trip_distance"] <= dc.TRIP_DISTANCE_MAX)
        mask_fare = (df["fare_amount"] >= dc.FARE_AMOUNT_MIN) & (df["fare_amount"] <= dc.FARE_AMOUNT_MAX)
        mask_tip = (df["tip_amount"] >= dc.TIP_AMOUNT_MIN) & (df["tip_amount"] <= dc.TIP_AMOUNT_MAX)
        mask_payment = df["payment_type"].isin(dc.PAYMENT_TYPE_ALLOWED)

        valid_mask = mask_passenger & mask_distance & mask_fare & mask_tip & mask_payment

        df_clean = df[valid_mask].copy()

        # 5) Stats
        clean_rows = len(df_clean)
        dropped_rows = raw_count - clean_rows
        cleaning_ratio = (dropped_rows / raw_count) if raw_count > 0 else 0.0

        stats = {
            "initial_rows": raw_count,
            "after_null_drop_rows": after_null_drop,
            "clean_rows": clean_rows,
            "dropped_rows": dropped_rows,
            "cleaning_ratio": cleaning_ratio,
            "violation_passenger": int((~mask_passenger).sum()),
            "violation_distance": int((~mask_distance).sum()),
            "violation_fare": int((~mask_fare).sum()),
            "violation_tip": int((~mask_tip).sum()),
            "violation_payment": int((~mask_payment).sum()),
        }

        logger.info(f"Cleaned data stats: {stats}")

        # 6) Save dropped sample to writable dir (fixes Airflow PermissionError)
        dropped_sample_path = None
        if dropped_rows > 0:
            try:
                self._ensure_artifact_dir()
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                dropped_sample_path = os.path.join(self.artifact_dir, f"dropped_data_sample_{ts}.csv")
                dropped_df = df[~valid_mask].head(100)
                dropped_df.to_csv(dropped_sample_path, index=False)
                logger.info(f"Saved dropped rows sample to: {dropped_sample_path}")
            except Exception as e:
                # Don't crash pipeline just because the sample can't be written;
                # main quality gates should still apply.
                logger.warning(f"Could not write dropped sample CSV: {e}")

        # Guards
        if clean_rows == 0:
            raise ValueError("Data Quality Critical: Resulting dataset is empty after cleaning.")

        if cleaning_ratio > dc.MAX_UNCLEAN_RATIO:
            error_msg = (
                f"Data Quality Breach! Removed {cleaning_ratio:.2%} of rows, "
                f"exceeding the limit of {dc.MAX_UNCLEAN_RATIO:.2%}. "
                f"Dropped sample: {dropped_sample_path or 'not written'}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Attach metadata for main.py (MLflow logging)
        df_clean.attrs["stats"] = stats
        df_clean.attrs["dropped_sample_path"] = dropped_sample_path

        return df_clean
