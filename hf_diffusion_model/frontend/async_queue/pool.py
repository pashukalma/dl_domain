# pool.py

import multiprocessing
import time

from server_block import gradio_demo

PROCESSES = multiprocessing.cpu_count() - 1

def run():
    print(f"Running with {PROCESSES} processes!")

    start = time.time()
    with multiprocessing.Pool(PROCESSES) as p:
        p.map_async(
            gradio_demo()
        )
        # clean up
        p.close()
        p.join()

    print(f"Time taken = {time.time() - start:.10f}")


if __name__ == "__main__":
    run()