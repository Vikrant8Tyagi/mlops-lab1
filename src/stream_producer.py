import json
import time
import logging
import pandas as pd
import numpy as np
from confluent_kafka import Producer
import src.data_contract as dc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CONF = {'bootstrap.servers': 'localhost:9092'}
TOPIC = "taxi_raw_events"
DATA_PATH = "data/green_tripdata_2023-01.parquet"
DELAY = 0.01  # Fast simulation (100 events/sec)


def delivery_callback(err, msg):
    if err:
        logger.error(f"Message failed delivery: {err}")


def run_producer():
    producer = Producer(CONF)

    logger.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)

    # Select columns from contract
    df = df[dc.REQUIRED_COLUMNS].copy()

    # Replace NaNs with None for valid JSON nulls
    df = df.replace({np.nan: None})

    records = df.to_dict(orient='records')
    logger.info(f"Starting stream of {len(records)} events...")

    try:
        for i, record in enumerate(records):
            # Serialize
            payload = json.dumps(record, default=str)

            # Produce
            producer.produce(
                TOPIC,
                value=payload,
                callback=delivery_callback
            )

            # Network Poll (batching optimization)
            if i % 100 == 0:
                producer.poll(0)
                logger.info(f"Sent {i} events...")

            time.sleep(DELAY)

    except KeyboardInterrupt:
        logger.info("Stopping producer...")
    finally:
        producer.flush()
        logger.info("Producer flushed and closed.")


if __name__ == "__main__":
    run_producer()
