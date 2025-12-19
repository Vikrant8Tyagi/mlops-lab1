import json
import logging
import signal
import time
from collections import deque
from confluent_kafka import Consumer, Producer
import mlflow
import src.data_contract as dc

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StreamProcessor")

# --- Configuration ---
KAFKA_CONF_C = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'taxi_feature_group',
    'auto.offset.reset': 'latest'
}
KAFKA_CONF_P = {'bootstrap.servers': 'localhost:9092'}

TOPIC_IN = "taxi_raw_events"
TOPIC_OUT = "taxi_features"

# MLflow Config
EXPERIMENT_NAME = "NYC_Taxi_Streaming"
METRIC_FLUSH_INTERVAL = 10  # seconds

# Circuit Breaker Config
WINDOW_SIZE = 1000
MAX_ERROR_RATE = dc.MAX_UNCLEAN_RATIO # Reuse contract threshold (0.15)

running = True

def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received.")
    running = False

class StreamContext:
    """Holds state for metrics and the circuit breaker."""
    def __init__(self):
        self.processed_count = 0
        self.drop_count = 0
        self.start_time = time.time()
        self.last_flush = time.time()

        # Sliding window of last N events (1=Success, 0=Fail)
        self.health_window = deque(maxlen=WINDOW_SIZE)

        # Detailed counters
        self.errors = {
            "schema_violation": 0,
            "value_violation": 0
        }

    def update(self, success: bool, reason: str = None):
        self.processed_count += 1
        self.health_window.append(1 if success else 0)

        if not success:
            self.drop_count += 1
            if reason:
                self.errors[reason] = self.errors.get(reason, 0) + 1

    def get_error_rate(self):
        if not self.health_window:
            return 0.0
        # sum(window) is count of 1s (successes)
        failures = len(self.health_window) - sum(self.health_window)
        return failures / len(self.health_window)

def validate_and_enrich(record: dict) -> dict:
    """
    Applies Data Contract rules.
    Raises ValueError for logic violations, TypeError for schema violations.
    """
    # 1. Schema/Type Check (Strict)
    try:
        # Explicit cast. If a field is missing, this raises KeyError/TypeError
        passenger_count = int(record['passenger_count'])
        trip_distance = float(record['trip_distance'])
        fare_amount = float(record['fare_amount'])
        tip_amount = float(record['tip_amount'])
        payment_type = float(record['payment_type'])
    except (KeyError, ValueError, TypeError):
        raise TypeError("Schema Violation: Missing or Malformed fields")

    # 2. Domain Validation (Contract)
    if not (dc.PASSENGER_MIN <= passenger_count <= dc.PASSENGER_MAX):
        raise ValueError("Value Violation: Passenger Count")

    if not (dc.TRIP_DISTANCE_MIN <= trip_distance <= dc.TRIP_DISTANCE_MAX):
        raise ValueError("Value Violation: Trip Distance")

    if not (dc.FARE_AMOUNT_MIN <= fare_amount <= dc.FARE_AMOUNT_MAX):
        raise ValueError("Value Violation: Fare Amount")

    if not (dc.TIP_AMOUNT_MIN <= tip_amount <= dc.TIP_AMOUNT_MAX):
        raise ValueError("Value Violation: Tip Amount")

    if payment_type not in dc.PAYMENT_TYPE_ALLOWED:
        raise ValueError("Value Violation: Payment Type")

    # 3. Feature Computation
    # Avoid div by zero
    record['tip_pct'] = tip_amount / (fare_amount + 0.0001)
    record['is_high_value'] = 1 if fare_amount > 20.0 else 0

    return record

def run():
    signal.signal(signal.SIGINT, signal_handler)

    consumer = Consumer(KAFKA_CONF_C)
    producer = Producer(KAFKA_CONF_P)
    consumer.subscribe([TOPIC_IN])

    ctx = StreamContext()

    # FIX: Explicitly set the tracking URI to ensure we talk to the Docker container
    mlflow.set_tracking_uri("http://localhost:5000")

    # Ensure experiment exists (useful if Docker was restarted and DB wiped)
    try:
        if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
            mlflow.create_experiment(EXPERIMENT_NAME)
    except Exception as e:
        logger.warning(f"Experiment creation warning: {e}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start a Long-Running MLflow Run
    with mlflow.start_run(run_name="stream_processor_v1"):
        logger.info(f"Stream Processor Active. Listening: {TOPIC_IN}")

        while running:
            msg = consumer.poll(1.0)
            if msg is None:
              continue
            if msg.error():
                logger.error(f"Consumer Error: {msg.error()}")
                continue

            try:
                payload = json.loads(msg.value().decode('utf-8'))

                # --- Core Logic ---
                enriched = validate_and_enrich(payload)

                # If we get here, it's valid
                producer.produce(TOPIC_OUT, value=json.dumps(enriched))
                ctx.update(success=True)

            except TypeError:
                ctx.update(success=False, reason="schema_violation")
            except ValueError:
                ctx.update(success=False, reason="value_violation")
            except json.JSONDecodeError:
                ctx.update(success=False, reason="malformed_json")

            producer.poll(0)

            # --- Observability Loop ---
            current_time = time.time()
            if current_time - ctx.last_flush > METRIC_FLUSH_INTERVAL:

                # 1. Throughput
                elapsed = current_time - ctx.start_time
                throughput = ctx.processed_count / elapsed if elapsed > 0 else 0

                # 2. Rolling Error Rate
                error_rate = ctx.get_error_rate()

                # 3. Log to MLflow
                metrics = {
                    "throughput_eps": throughput,
                    "rolling_error_rate": error_rate,
                    "total_processed": ctx.processed_count,
                    "total_dropped": ctx.drop_count,
                    "schema_errors": ctx.errors["schema_violation"],
                    "value_errors": ctx.errors["value_violation"]
                }
                mlflow.log_metrics(metrics, step=ctx.processed_count)

                logger.info(f"Heartbeat: {throughput:.1f} events/sec | Error Rate: {error_rate:.2%}")

                # 4. Safety Breaker Check
                if error_rate > MAX_ERROR_RATE and ctx.processed_count > 100:
                    logger.critical(f"🚨 CIRCUIT BREAKER TRIPPED! Error rate {error_rate:.2%} > {MAX_ERROR_RATE:.2%}")
                    # In production, we might stop the consumer or trigger a PagerDuty alert
                    # For lab, we just log visibly.

                ctx.last_flush = current_time

    consumer.close()
    producer.flush()

if __name__ == "__main__":
    run()
