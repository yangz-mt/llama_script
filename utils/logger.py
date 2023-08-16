import os
import logging
import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class Logger:
    def __init__(self, log_path='', rank=None, cuda=False, debug=False):
        self.logger = logging.getLogger(__name__)
        self.cuda = cuda
        self.rank = rank
        # self.log_path = log_path + '.txt'
        self.log_path = log_path + '/' + self.get_latest_file()
        self.debug = debug

    def info(self, message, log_=True, print_=True, *args, **kwargs):
        if self.rank == 0:
            if print_:
                self.logger.info(message, *args, **kwargs)

            if log_:
                with open(self.log_path, "a+") as f_log:
                    f_log.write(message + "\n")

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)

    def get_latest_file(self, file_path='./error_log'):
        file_list = os.listdir(file_path)
        print(file_list)
        file_list.sort(key=lambda fn: os.path.getmtime(file_path + "/" + fn))
        file_new = file_list[-1]
        print(f'using {file_new} log')
        return file_new

    @property
    def log_file_path(self):
        return self.log_path
    
# logger = Logger()
# print(logger.get_latest_file())
    
