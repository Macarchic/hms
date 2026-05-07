"""
Centralized logging: INFO+ goes to console, DEBUG+ goes to file.

Usage:
    from logger import get_logger
    log = get_logger('spec', log_dir='/path/to/logs')

    log.info('epoch done')      # console + file
    log.debug('batch 3/100')    # file only
"""

import logging
import os


def get_logger(name: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt_file    = logging.Formatter('%(asctime)s  %(levelname)-5s  %(message)s',
                                    datefmt='%H:%M:%S')
    fmt_console = logging.Formatter('%(message)s')

    # file: everything (DEBUG+)
    fh = logging.FileHandler(os.path.join(log_dir, f'{name}_detail.log'), mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)

    # console: only INFO+
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
