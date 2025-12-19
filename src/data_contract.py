"""
src/data_contract.py

Single Source of Truth for Data Quality Rules.
"""

# Versioning allows us to track which rules were active
# for a specific model run.
CONTRACT_VERSION = "1.0.0"

# -------------------------------------------------------------------
# System Limits
# -------------------------------------------------------------------
# If >15% of data is malformed, we assume a systemic upstream failure
# (e.g., broken sensor).
# Increased from 5% to 15% after observing real-world noise
# in Jan 2023 data.
MAX_UNCLEAN_RATIO = 0.15


# -------------------------------------------------------------------
# Schema Definition
# -------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "payment_type",
]


# -------------------------------------------------------------------
# Domain Rules (Inclusive Boundaries)
# -------------------------------------------------------------------
# Passenger Count: Physical constraints of a vehicle.
PASSENGER_MIN = 1
PASSENGER_MAX = 10

# Trip Distance: Minimum to register the meter,
# maximum for a single shift.
TRIP_DISTANCE_MIN = 0.1
TRIP_DISTANCE_MAX = 1000.0

# Financials: Fares must be positive.
# Cap at $5000 to catch outliers or errors.
FARE_AMOUNT_MIN = 2.5
FARE_AMOUNT_MAX = 5000.0

TIP_AMOUNT_MIN = 0.0
TIP_AMOUNT_MAX = 2000.0


# -------------------------------------------------------------------
# Categorical Rules
# -------------------------------------------------------------------
# Payment Types:
# 1 = Credit
# 2 = Cash
# 3 = No Charge
# 4 = Dispute
# 5 = Unknown
# 6 = Voided
#
# Defined as floats to align with pandas nullable integer behavior.
PAYMENT_TYPE_ALLOWED = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
