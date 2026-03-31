import logging
import os


LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class ColoredFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        reset = "\033[0m"
        colors = {
            logging.DEBUG: f"{reset}\033[36m",
            logging.INFO: f"{reset}\033[32m",
            logging.WARNING: f"{reset}\033[33m",
            logging.ERROR: f"{reset}\033[31m",
            logging.CRITICAL: f"{reset}\033[35m",
        }
        fmt = "{color}%(levelname)s %(asctime)s %(process)d %(filename)s:%(lineno)4d:{reset} %(message)s"
        self._formatters = {
            level: logging.Formatter(fmt.format(color=color, reset=reset))
            for level, color in colors.items()
        }
        self._default_formatter = self._formatters[logging.INFO]

    def format(self, record):
        formatter = self._formatters.get(record.levelno, self._default_formatter)
        return formatter.format(record)


def get_logger(name, level=logging.INFO):
    env_level = os.environ.get("LOG_LEVEL")
    if env_level is not None:
        env_level = env_level.upper()
        if env_level not in LOG_LEVELS:
            raise ValueError(
                f"Invalid LOG_LEVEL: {env_level}, expected one of {list(LOG_LEVELS)}"
            )
        level = LOG_LEVELS[env_level]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
    return logger
