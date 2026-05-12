from colorama import Fore, Back, Style
from icecream import ic

       
default_output = ic.outputFunction
def _ic_output_function(message):
    print(f'\n{Back.LIGHTRED_EX}{Fore.BLACK} DEBUG {Style.RESET_ALL} ')
    print(f'{Fore.YELLOW}{message}{Style.RESET_ALL}')
ic.configureOutput(
    outputFunction=_ic_output_function,
    prefix='',
)

def notice(message, color = "green"):
    if color == "green":
        print(f'\n{Back.LIGHTGREEN_EX}{Fore.BLACK} {message} {Style.RESET_ALL} ')
    elif color == "red":
        print(f'\n{Back.LIGHTRED_EX}{Fore.BLACK} {message} {Style.RESET_ALL} ')
    elif color == "blue":
        print(f'\n{Back.LIGHTBLUE_EX}{Fore.BLACK} {message} {Style.RESET_ALL} ')

def warn(message, color = "yellow", end="\n"):
    if color == "yellow":
        print(f'{Fore.YELLOW}{message}{Style.RESET_ALL}', end=end)
    elif color == "red":
        print(f'{Fore.RED}{message}{Style.RESET_ALL}', end=end)

import time
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' completed in {elapsed_time:.2f} s.")
        return result
    return wrapper


import sys
from contextlib import contextmanager
from functools import wraps

@contextmanager
def _tee_output(filename):
                     
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for file in self.files:
                file.write(text)
                file.flush()
        def flush(self):
            for file in self.files:
                file.flush()
    with open(filename, 'a', encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        try:
            yield
        finally:
            sys.stdout = original_stdout

def log_to_file(filename):
                         
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with _tee_output(filename):
                return func(*args, **kwargs)
        return wrapper
    return decorator
