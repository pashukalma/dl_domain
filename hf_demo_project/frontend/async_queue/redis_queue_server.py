# redis_queue_server.py
import multiprocessing
from redis_queue_worker import worker

PROCESSES = 4

def run():
    processes = []
    print(f"Running with {PROCESSES} processes!")
    while True:
        for w in range(PROCESSES):
            p = multiprocessing.Process(target=worker)
            processes.append(p)
            p.start()
        for p in processes:
            p.join()


if __name__ == "__main__":
    run()