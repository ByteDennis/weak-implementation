from typing import Optional, Union, Literal
from upath import UPath
from tqdm import tqdm
from loguru import logger as llogger

GLOBAL = '__GLOBAL__'

class TqdmLogging:
    """Custom handler to make loguru compatible with tqdm progress bars"""

    def __init__(self, level: str = "INFO"):
        self.level = level

    def write(self, message):
        # Remove the trailing newline that loguru adds
        message = message.rstrip("\n")
        if message:
            tqdm.write(message)

    def flush(self):
        pass


class Logging:
    """Experiment logging using loguru with tqdm compatibility"""

    def __init__(
        self,
        run_id: Optional[str] = None,
        level: Literal['INFO', 'DEBUG', 'ERROR'] = "INFO",
        format_str: Optional[str] = None,
        log_dir: Optional[str| UPath] = None,
        disable_color: bool = True,
    ):
        # Remove default logger ...
        llogger.remove()

        self.run_id = run_id or __name__
        self.level = level.upper()
        self.log_dir = log_dir and UPath(log_dir)
        self.disable_color = disable_color

        # Format related ...
        if format_str is None:
            if disable_color:
                self.format_str = (
                    "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
                )
            else:
                self.format_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> | {message}"
        else:
            self.format_str = format_str
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup tqdm compatible handlers"""
        tqdm_handler = TqdmLogging(self.level)
        llogger.add(
            tqdm_handler,
            format=self.format_str,
            level=self.level,
            colorize=not self.disable_color,
            backtrace=True,
            diagnose=True,
        )

        # File handler (always without tqdm)
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_dir / f"{self.run_id}.log"
            llogger.add(
                log_file,
                format=self.format_str,
                level=self.level,
                rotation="100 MB",
                retention=None,
                compression="zip",
                colorize=False,
                backtrace=True,
                diagnose=True,
            )
    
    def get_logger(self, name: Optional[str] = None):
        """Get logger instance with optional name"""
        if name:
            return llogger.bind(name=name)
        return llogger.bind(name=self.run_id)

    def exp_start(self, **things_to_log):
        """Log experiment start with parameters"""
        exp_logger = self.get_logger()
        exp_logger.info("=" * 50)
        exp_logger.info(f"Starting experiment: {self.run_id}")
        for key, value in things_to_log.items():
            exp_logger.info(f"{key}: {value}")
        exp_logger.info("=" * 50)

    def exp_end(self, **things_to_log):
        """Log experiment end with results"""
        exp_logger = self.get_logger()
        exp_logger.info("=" * 50)
        exp_logger.info(f"Experiment {self.run_id} completed")
        for key, value in things_to_log.items():
            exp_logger.info(f"{key}: {value}")
        exp_logger.info("=" * 50)

def setup_logger(
    run_id: Optional[str] = None,
    level: str = "INFO",
    disable_color: bool = True,
    log_dir: Optional[str] = None,
    reset_global: bool = False
) -> Logging:
    global logger
    if reset_global and logger is not None:
        logger.remove(lambda record: record["extra"].get("run_id") == GLOBAL)
        return setup_logger(GLOBAL, level, disable_color, log_dir)
    return Logging(
        run_id=run_id, level=level, disable_color=disable_color, log_dir=log_dir
    ).get_logger(run_id)

def get_logger(name: Optional[str] = None):
    if not llogger._core.handlers:
        raise RuntimeError(
            "Logger not configured! Call setup_logger() first:\n"
            "logger = setup_logger(run_id='my_experiment')"
        )
    return llogger.bind(name=name or __name__)

logger = setup_logger('__GLOBAL__', disable_color=False)

__all__ = ['Logging', 'setup_logger', 'get_logger', 'logger']