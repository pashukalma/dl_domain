#  simple_task_queue_logging_separate_files.py
import logging
import multiprocessing
import os
import time

from server_block import gradio_demo

PROCESSES = multiprocessing.cpu_count() - 1
NUMBER_OF_TASKS = 10

def create_logger(pid):
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"logs/process_{pid}.log")
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def process_tasks(task_queue):
    proc = os.getpid()
    logger = create_logger(proc)
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
    run()
