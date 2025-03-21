from pathlib import Path
from datetime import datetime
import logging 
from logging.handlers import RotatingFileHandler
import os 

# we will create a default log dir in the root of the project 
Log_dir = Path(__file__).resolve().parents[1] / 'logs'
def setup_logger(name: str, log_dir: Path = Log_dir) -> logging.logger:
    """
    a function that sets up a logger for each module in a organized way and return a logger instance
    
    Args:
        - name: the name of the logger 
        - log_dir: the directory where the log file will be saved
    """
    
    # create the log directory if it does not exist 
    os.makedirs(log_dir, exist_ok=True)
    
    #create an instance of the logger 
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # set the log level     
        logger.setLevel(logging.INFO)
        
        # create a good formatter for easy reading 
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt = '%Y-%m-%d %H:%M:%S'
        )
        
        # create a file handler
        file_handler = RotatingFileHandler(
            filename = log_dir / f'{name}.log',
            maxBytes=1024 * 1024 * 2, # 2MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # create a stream handler 
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        
        # add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
    return logger