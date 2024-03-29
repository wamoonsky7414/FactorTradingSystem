import os
import json
import pandas as pd
from IPython.display import display
import tejapi

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .tej_handler import TEJHandler

class TWMarketMonitor(TEJHandler):
    def __init__(self):
        super().__init__()

