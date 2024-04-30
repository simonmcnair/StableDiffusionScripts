import logging
import time
import os
import platform
from pathlib import Path
import functools

_logger = None

def setup_logging(log_file, errorlog_path,log_level='debug'):

    log_level_map = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING,'error': logging.ERROR}

    log_level = log_level.lower()
    if log_level not in log_level_map:
        print(f"Invalid log level: {log_level}. Defaulting to 'debug'.")
        log_level = 'debug'

    console_log_level = log_level_map['info']

    try:
            logger = logging.getLogger()
            if not logger.handlers:
                # Create a handler and set the level to the lowest level you want to log
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

                filehandler = logging.FileHandler(log_file)
                filehandler.setLevel(log_level_map[log_level])# file log level assigned elsewhere
                filehandler.setFormatter(formatter)
                logger.addHandler(filehandler)

                error_handler = logging.FileHandler(errorlog_path)
                error_handler.setLevel(log_level_map['error'])  # Only logs messages with ERROR level or higher
                error_handler.setFormatter(formatter)
                logger.addHandler(error_handler)

                console_handler = logging.StreamHandler()
                console_handler.setLevel(console_log_level) # log debug to console
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

                logger.setLevel(logging.DEBUG)
                print("Handlers are set up.")

            else:
                print("Handlers are already setup")

            return logger

    except Exception as e:
        print(f"An error occurred during logging setup: {e}.  Press any key to continue")

def get_logger(logfile,errorlogfile):
    global _logger
    if _logger is None:
        #_logger = setup_logging()

        _logger = setup_logging(logfile, errorlogfile, log_level='info')

    return _logger

def timing_decorator(func):
    #global timing_debug

    def wrapper(*args, **kwargs):
        #if timing_debug == False:
        #    return
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' took {end_time - start_time} seconds to execute.")
        return result
    return wrapper

def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling function {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

def get_operating_system():
    system = platform.system()
    return system

def get_script_name(mypath):
    # Use os.path.basename to get the base name (script name) from the full path
    #basename = os.path.basename(path)
    return Path(mypath).stem
    #return os.path.basename(__file__)

def get_script_path(mypath):
    return os.path.dirname(os.path.realpath(mypath))

#_logger = None
#logger = get_logger(__file__,__file__)


logger = get_logger(__name__ + '.log',__name__ + '_error.log')
