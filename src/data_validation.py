import logging
import pandas as pd
import great_expectations as gx
import great_expectations.expectations as gxe
from great_expectations.core.expectation_suite import ExpectationSuite
import src.data_contract as dc

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Ephemeral Context: In-memory configuration suitable for automated pipelines.
        self.context = gx.get_context(mode="ephemeral")
        self.datasource_name = "pandas_datasource"
        self.asset_name = "taxi_dataframe"
        self.suite_name = "taxi_quality_suite"
        self.validation_results = None

    def validate(self) -> bool:
        logger.info("Validating data with Great Expectations (v1.x)...")

        # 1. Setup Datasource (Idempotent)
        try:
            ds = self.context.data_sources.get(self.datasource_name)
        except KeyError:
            ds = self.context.data_sources.add_pandas(self.datasource_name)

        try:
            asset = ds.get_asset(self.asset_name)
        except LookupError:
            asset = ds.add_dataframe_asset(name=self.asset_name)

        # 2. Create Expectation Suite
        suite = ExpectationSuite(name=self.suite_name)

        # --- Rule A: Structural Integrity ---
        suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(min_value=1))
        for col in dc.REQUIRED_COLUMNS:
            suite.add_expectation(gxe.ExpectColumnToExist(column=col))

        # --- Rule B: Semantic Domains (Contract Enforcement) ---
        # Added mostly=0.99 to allow up to 1% exceptions per column
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(
                column="passenger_count",
                min_value=dc.PASSENGER_MIN,
                max_value=dc.PASSENGER_MAX
            )
        )

        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(
                column="trip_distance",
                min_value=dc.TRIP_DISTANCE_MIN,
                max_value=dc.TRIP_DISTANCE_MAX
            )
        )

        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(
                column="fare_amount",
                min_value=dc.FARE_AMOUNT_MIN,
                max_value=dc.FARE_AMOUNT_MAX
            )
        )

        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(
                column="tip_amount",
                min_value=dc.TIP_AMOUNT_MIN,
                max_value=dc.TIP_AMOUNT_MAX
            )
        )

        # --- Rule C: Categorical & Type Safety ---
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(
                column="payment_type", value_set=dc.PAYMENT_TYPE_ALLOWED
            )
        )

        suite.add_expectation(
            gxe.ExpectColumnValuesToBeOfType(
                column="trip_distance", type_="float"
            )
        )

        # 3. Run Validation
        batch_def = asset.add_batch_definition_whole_dataframe("whole_df")
        batch = batch_def.get_batch(batch_parameters={"dataframe": self.df})
        self.validation_results = batch.validate(suite)

        # 4. Result Handling
        if not self.validation_results.success:
            logger.error("❌ GX VALIDATION FAILED!")
            for res in self.validation_results.results:
                if not res.success:
                    col = res.expectation_config.kwargs.get("column", "Table-Level")
                    type_ = res.expectation_config.type
                    logger.error(f"   - Violation: {col} | Rule: {type_}")

            raise ValueError("Critical Data Validation Failed. Check MLflow artifacts for details.")

        logger.info("✅ Great Expectations passed.")
        return True
