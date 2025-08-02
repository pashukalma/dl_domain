# redis_queue_client.py
import redis

from redis_queue import SimpleQueue
from server_block import gradio_demo

NUMBER_OF_TASKS = 10

if __name__ == "__main__":
    r = redis.Redis()
    queue = SimpleQueue(r, "sample")
    count = 0
    for num in range(NUMBER_OF_TASKS):
        queue.enqueue(gradio_demo(),num)
        count += 4
    print(f"Enqueued {count} tasks!")