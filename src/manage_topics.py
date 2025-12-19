from confluent_kafka.admin import AdminClient, NewTopic
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KafkaAdmin")

CONF = {'bootstrap.servers': 'localhost:9092'}

def create_topics():
    admin_client = AdminClient(CONF)

    topics = [
        NewTopic("taxi_raw_events", num_partitions=2, replication_factor=1),
        NewTopic("taxi_features", num_partitions=2, replication_factor=1)
    ]

    # Explicitly create topics
    fs = admin_client.create_topics(topics)

    for topic, f in fs.items():
        try:
            f.result()  # Wait for result
            logger.info(f"Topic '{topic}' created successfully.")
        except Exception as e:
            logger.warning(
                f"Topic '{topic}' creation failed (might already exist): {e}"
            )

if __name__ == "__main__":
    create_topics()
