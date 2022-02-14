from functools import wraps
import time
import logging

logger = logging.getLogger("DCA_info_logger")
time_logger = logging.getLogger("DCA_time_logger")


def logger_time(func):
    """Prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(
            "Function {0} executed in {1:f} s".format(func.__name__, end - start)
        )
        time_logger.debug(
            "Elapsed time: {0}-{1:f} s".format(func.__name__, end - start)
        )
        return result

    return wrapper


def get_parameters(class_name):
    attr_names = [
        a
        for a in dir(class_name)
        if not a.startswith("__") and not callable(getattr(class_name, a))
    ]
    param_dict = {}
    for k in attr_names:
        value = getattr(class_name, k)
        if isinstance(value, (int, str, float, list, bool)):
            param_dict[k] = value
        else:
            pass
    return param_dict
