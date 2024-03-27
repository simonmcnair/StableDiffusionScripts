import logging

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
