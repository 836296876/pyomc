from time import time

def timer(func):
    def func_wrapper(*args, **kwargs):
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print(f"{func.__name__}执行完毕，用时{time_spend}s.")
        return result
    return func_wrapper
