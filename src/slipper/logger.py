import sys
import warnings

from loguru import logger
from tqdm import tqdm

NAME = "<blue>Slipper</blue>"
TIME = "{time:DD/MM HH:mm:ss}"
FMT = f"|{NAME}|{TIME}|" "{level}| <green>{message}</green>"

logger.remove(0)
logger.configure(
    handlers=[
        dict(
            sink=lambda msg: tqdm.write(msg, end=""),
            format=FMT,
            colorize=True,
            level="INFO",
        )
    ]
)
#
# logger.remove()
# logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [<level>{level: ^12}</level>] <level>{message}</level>"
# logger.configure(handlers=[dict(sink=lambda msg: tqdm.write(msg, end=''), format=logger_format, colorize=True)])
# logger.add(
#     sys.stderr,
#     format=(f"|{NAME}|{TIME}|" "{level}| <green>{message}</green>"),
#     colorize=True,
#     level="INFO",
# )
#
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# import logging
# NAME = "Slipper"
# TIME = "{time:DD/MM HH:mm:ss}"
# FMT = '<blue>%(name)s<blue> | %(levelname)s | <green> %(message)s </green>'
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
#
# logger = logging.getLogger(NAME)
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter(FMT, datefmt="%H:%M:%S")
# handler.setFormatter(formatter)
# logger.addHandler(handler)
