
import os
import random
import logging
import time
import inspect

def set_seeds(seed:int):
    """sets the seeds for various functions: random, os, numpy, torch

    Args:
        seed (int)
    """    
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass  # PyTorch is not installed; skip

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True) may reduce the number of operations usable
    except ImportError:
        pass

def get_logger(loggingLevel=logging.WARNING, filename=None):
    logging.basicConfig(level=logging.INFO, filename=filename)
    logger = logging.getLogger(__name__)

    return logger


def numberToBase(n, b)->str:
    def intToDigit(i):
        if i < 10:
            return str(i)
        elif i < 36:
            return chr(ord('A') + i - 10)
        elif i < 62:
            return chr(ord('a') + i - 36)
        else:
            raise ValueError(f'base is too large, maximum b supported is 62 to prevent the use of non alphanumeric symbols')
    assert b <= 62, f"max supported base is 62, given: {b}"
    if n == 0:
        return '0'
    digits = ''
    while n:
        digits += intToDigit(n % b)
        n //= b
    return digits[::-1]

def revertNumberToBase(digits, b)->int:
    def digitToInt(d):
        try:
            return int(d)
        except ValueError:
            pass
        if d >= 'A' and d <='Z':
            return ord(d) - ord('A') + 10
        elif d >= 'a' and d <='z':
            return ord(d) - ord('a') + 36
        else:
            raise ValueError(f'only alphanumeric digits are supported, given: {d}')
    assert b <= 62, f"max supported base is 62, given: {b}"
    n = 0
    for d in digits:
        n *= b
        n += digitToInt(d)
    return n

def get_time_UID(high_precision=True):
    """return a universal string id based on time

    Args:
        high_precision (bool, optional): flag to use a high precision to use 1 micro-s resolution in place of 1s, this will likely guarantee different timestamps even for consecutive runs. Defaults to True.

    Returns:
        _type_: _description_
    """
    t = time.time()
    m = 1000000 if high_precision else 1
    return numberToBase(int(t*m) ,62)

def derive_time_from_UID(uid:str, high_precision=True):
    d =  1000000 if high_precision else 1
    return revertNumberToBase(uid, 62)/d

def get_abs_class_file_path(class_obj):
    return os.path.dirname(inspect.getfile(class_obj.__class__))
