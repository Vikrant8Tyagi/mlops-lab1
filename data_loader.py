import logging
import pandas as pd
import src.data_contract as dc

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.path}...")

        # --------------------------------------------------
        # Load Data (Fail Fast)
        # --------------------------------------------------
        try:
            df = pd.read_parquet(self.path)
        except Exception as e:
            logger.error(f"Failed to read parquet file: {e}")
            raise

        # --------------------------------------------------
        # 1. Critical Schema Check
        # --------------------------------------------------
        missing_cols = [c for c in dc.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Schema Violation: Missing columns {missing_cols}"
            )

        initial_count = len(df)
        logger.info(f"Initial row count: {initial_count}")

        # --------------------------------------------------
        # 2. Type Enforcement
        # --------------------------------------------------
        if "payment_type" in df.columns:
            df["payment_type"] = pd.to_numeric(
                df["payment_type"], errors="coerce"
            )

        # --------------------------------------------------
        # 3. Drop Nulls
        # --------------------------------------------------
        df = df.dropna(subset=dc.REQUIRED_COLUMNS)

        # --------------------------------------------------
        # 4. Validation Masks (Detailed Observability)
        # --------------------------------------------------
        mask_passenger = (
            (df["passenger_count"] >= dc.PASSENGER_MIN)
            & (df["passenger_count"] <= dc.PASSENGER_MAX)
        )

        mask_distance = (
            (df["trip_distance"] >= dc.TRIP_DISTANCE_MIN)
            & (df["trip_distance"] <= dc.TRIP_DISTANCE_MAX)
        )

        mask_fare = (
            (df["fare_amount"] >= dc.FARE_AMOUNT_MIN)
            & (df["fare_amount"] <= dc.FARE_AMOUNT_MAX)
        )

        mask_tip = (
            (df["tip_amount"] >= dc.TIP_AMOUNT_MIN)
            & (df["tip_amount"] <= dc.TIP_AMOUNT_MAX)
        )

        mask_payment = df["payment_type"].isin(dc.PAYMENT_TYPE_ALLOWED)

        # Combined mask (ALL rules must pass)
        valid_mask = (
            mask_passenger
            & mask_distance
            & mask_fare
            & mask_tip
            & mask_payment
        )

        df_clean = df[valid_mask]

        # --------------------------------------------------
        # 5. Threshold Enforcement & Reporting
        # --------------------------------------------------
        filtered_count = len(df_clean)
        removed_count = initial_count - filtered_count
        cleaning_ratio = (
            removed_count / initial_count if initial_count > 0 else 0
        )

        stats = {
            "initial_rows": initial_count,
            "clean_rows": filtered_count,
            "dropped_rows": removed_count,
            "cleaning_ratio": cleaning_ratio,
            "violation_passenger": (~mask_passenger).sum(),
            "violation_distance": (~mask_distance).sum(),
            "violation_fare": (~mask_fare).sum(),
            "violation_tip": (~mask_tip).sum(),
            "violation_payment": (~mask_payment).sum(),
        }

        logger.info(f"Cleaned data stats: {stats}")

        # --------------------------------------------------
        # 6. Save Debug Artifacts (Inspectable Safety Breaker)
        # --------------------------------------------------
        if removed_count > 0:
            dropped_df = df[~valid_mask].head(100)
            dropped_df.to_csv(
                "dropped_data_sample.csv", index=False
            )

        # --------------------------------------------------
        # Guards
        # --------------------------------------------------
        if filtered_count == 0:
            raise ValueError(
                "Data Quality Critical: Dataset is empty after cleaning."
            )

        if cleaning_ratio > dc.MAX_UNCLEAN_RATIO:
            error_msg = (
                f"Data Quality Breach! Removed {cleaning_ratio:.2%} of rows, "
                f"exceeding the limit of {dc.MAX_UNCLEAN_RATIO:.2%}. "
                "Check 'dropped_data_sample.csv'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Attach metadata for MLflow logging
        df_clean.attrs["stats"] = stats

        return df_clean
