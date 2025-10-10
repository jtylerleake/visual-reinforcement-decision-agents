
from common.modules import os, sys, dt, time, Optional, logging, handlers

PARENT_DIR = "experiment-configs"
EXPERIMENT_LOGGER_REGISTRY = {}


class ExperimentLogger:
    
    """
    Centralized logging system for experiments. Log files are saved to their 
    relevant experiment directories under experiment-configs/[experiment-name]
    """
    
    def __init__(self, name: str, save_log_file: Optional[bool] = True):

        self.name = name
        self.save_log_file = save_log_file
        self.logger = logging.getLogger(name)
        
        # set log level and setup handlers
        log_level = 'INFO'
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        
        # console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # file handler
        if self.save_log_file:
            
            # create experiment-specific log file path
            log_dir = os.path.join(PARENT_DIR, self.name, "experiment-logs")
            os.makedirs(log_dir, exist_ok=True)  # ensure directory exists
            
            # timestamp log file
            timestamp = dt.now().strftime("%Y-%m-%d__%H.%M.%S")
            log_file = os.path.join(log_dir, f"{timestamp}-execution-log.txt")
            
            # Rotating file handler
            file_handler = handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10mb
                backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
    def clean_message(self, message:str):
        message.replace('\n', ' ')
        message = ' '.join(message.split())
        return message
    
    def debug(self, message: str):
        self.logger.debug(self.clean_message(message))
        
    def info(self, message: str): 
        self.logger.info(self.clean_message(message))
    
    def warning(self, message: str):
        self.logger.warning(self.clean_message(message))
    
    def error(self, message: str):
        self.logger.error(self.clean_message(message))
    
    def critical(self, message: str):
        self.logger.critical(self.clean_message(message))
        
    def h1(self, message):
        self.logger.info("-"*70)
        self.logger.info(self.clean_message(message))
        self.logger.info("-"*70)
        
    def h2(self, message):
        self.logger.info("-"*35)
        self.logger.info(self.clean_message(message))
        self.logger.info("-"*35)


def get_logger(experiment_name, save_log_file = True):
    """Returns the ExperimentLogger instance for the given experiment_name.
    If it does not exist, creates and stores a new one"""
    if experiment_name not in EXPERIMENT_LOGGER_REGISTRY:
        from src.utils.system_logging import ExperimentLogger
        EXPERIMENT_LOGGER_REGISTRY[experiment_name] = \
            ExperimentLogger(experiment_name, save_log_file=save_log_file)
    return EXPERIMENT_LOGGER_REGISTRY[experiment_name]


def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("decorator")
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned {result}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper


def log_execution_time(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("decorator")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {e}")
            raise
    return wrapper
