"""
This script contains functions for log recording from the standard output.
"""
import sys
import os

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def make_logs(log_path, log_file, err_file):
    # make log files with log_path, log_file and err_file
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_filepath = log_path + log_file
    err_filepath = log_path + err_file
    sys.stdout = Logger(log_filepath)
    sys.stderr = Logger(err_filepath)