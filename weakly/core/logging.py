from typing import Dict, List
from loguru import logger
from tqdm import tqdm


class ExperimentLogger:
    # Centralized logging for all experiment components

    def __init__(self):
        self._logger_configured = False

    def setup_logging(self, run_name: str, log_level: str = "INFO"):
        # Configures logging to file and console
        if self._logger_configured:
            logger.remove()
        # Remove default handler
        logger.remove()
        # Add console handler with tqdm compatibility
        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            format="{time:YYYY-MM-DD HH:mm:ss} - {message}",
            level=log_level,
            colorize=True,
        )
        # Add file handler
        logger.add(
            f"logs/{run_name}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
            level=log_level,
            rotation="10 MB",
            retention="7 days",
        )
        self._logger_configured = True
        logger.info(f"Logging configured for experiment: {run_name}")

    def log_experiment_start(self, config: Dict):
        logger.info("=" * 50)
        logger.info("EXPERIMENT STARTED")
        logger.info("=" * 50)

        for key, value in config.items():
            logger.info(f"Config - {key}: {value}")
        logger.info("=" * 50)

    def log_model_performance(self, model_name: str, metrics: Dict):
        # Logs model evaluation metrics
        logger.info(f"Model Performance - {model_name}")
        logger.info("-" * 30)
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                logger.info(f"{metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"{metric_name}: {metric_value}")
        logger.info("-" * 30)

    def log_error(self, error: Exception, context: str):
        # Logs errors with full context for debugging
        logger.error(f"Error in {context}")
        logger.error(f"Error type: {type(error).__name__}")
        logger.error(f"Error message: {str(error)}")
        logger.exception("Full traceback:")

    def log_progress(self, message: str, step: int = None, total: int = None):
        # Logs progress updates
        if step is not None and total is not None:
            logger.info(f"Progress [{step}/{total}]: {message}")
        else:
            logger.info(f"Progress: {message}")

    def log_checkpoint(self, checkpoint_path: str, epoch: int, metrics: Dict = None):
        # Logs model checkpoint information
        logger.info(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")
        if metrics:
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value}")

    def log_experiment_end(self, status: str = "completed"):
        # Logs experiment completion
        logger.info("=" * 50)
        logger.info(f"EXPERIMENT {status.upper()}")
        logger.info("=" * 50)
