# src/stream_validator.py
from confluent_kafka import Consumer
import json

c = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'manual_verifier',
    'auto.offset.reset': 'latest'
})
c.subscribe(['taxi_features'])

print("👀 Watching 'taxi_features' topic...")
try:
    while True:
        msg = c.poll(1.0)
        if msg is None:
          continue
        data = json.loads(msg.value().decode('utf-8'))
        print(f"✅ Enriched Event: Fare=${data['fare_amount']} | HighVal={data['is_high_value']}")
except KeyboardInterrupt:
    pass
finally:
    c.close()
