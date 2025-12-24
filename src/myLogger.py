import os
import os.path as osp
import logging

class myLogger:

    def __init__(self, logger_name, config):
        self.logger_name = logger_name
        self.config = config

    def get_logger(self):
        os.makedirs(self.config.logs_path, exist_ok=True)
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.INFO)
        file_hand = logging.FileHandler(osp.join(self.config.logs_path, f"{self.config.log_filename}.log"))
        file_hand.setLevel(logging.INFO)   
        cons_hand = logging.StreamHandler()
        cons_hand.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_hand.setFormatter(formatter)
        cons_hand.setFormatter(formatter)

        logger.addHandler(file_hand)
        logger.addHandler(cons_hand)

        return logger