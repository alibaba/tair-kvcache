import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.propagate = False
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[KVCM] %(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s", "%m-%d %H:%M:%S")
handler.setFormatter(formatter)

logger.addHandler(handler)
