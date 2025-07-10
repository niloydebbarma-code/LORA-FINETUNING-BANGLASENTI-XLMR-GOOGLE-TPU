import logging
import os
from loguru import logger
from rich.logging import RichHandler

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

def setup_logging(log_level=logging.DEBUG, log_file=None):
    os.makedirs('logs', exist_ok=True)
    logger.remove()
    # Detailed format for analysis
    analysis_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{process}</cyan>:<cyan>{thread}</cyan> | "
        "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>\n{exception}"
    )
    if log_file:
        logger.add(log_file, rotation="10 MB", retention="5 days", level="DEBUG",
                   format=analysis_format, enqueue=True, backtrace=True, diagnose=True, catch=True)
    # Log to console with rich (INFO+ for readability)
    logger.add(RichHandler(rich_tracebacks=True, markup=True, show_time=True, show_level=True, show_path=False),
               level=log_level, enqueue=True)
    # Intercept standard logging and route to loguru
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(log_level)
    # Silence noisy libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.WARNING)
