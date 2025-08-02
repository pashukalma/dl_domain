# simple_task_queue_logging.py

import logging
import multiprocessing
import os
import time

from server_block import gradio_demo

PROCESSES = multiprocessing.cpu_count() - 1
NUMBER_OF_TASKS = 10

def process_tasks(task_queue):
    logger = multiprocessing.get_logger()
    proc = os.getpid()
    while not task_queue.empty():
        try:
            gradio_demo()
        except Exception as e:
            logger.error(e)
        logger.info(f"Process {proc} completed successfully")
    return True


def add_tasks(task_queue, number_of_tasks):
    for num in range(number_of_tasks):
        task_queue.put(num)
    return task_queue


def run():
    empty_task_queue = multiprocessing.Queue()
    full_task_queue = add_tasks(empty_task_queue, NUMBER_OF_TASKS)
    processes = []
    print(f"Running with {PROCESSES} processes!")
    start = time.time()
    for w in range(PROCESSES):
        p = multiprocessing.Process(target=process_tasks, args=(full_task_queue,))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print(f"Time taken = {time.time() - start:.10f}")


if __name__ == "__main__":
    multiprocessing.log_to_stderr(logging.ERROR)
    run()