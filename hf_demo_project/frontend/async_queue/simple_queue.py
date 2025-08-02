# simple_queue.py
import multiprocessing

def run():
    queue = multiprocessing.Queue()

    print("Enqueuing...")
    for idx in range(10):
        queue.put(idx)

    print("\nDequeuing...")
    while not queue.empty():
        print(queue.get())


if __name__ == "__main__":
    run()