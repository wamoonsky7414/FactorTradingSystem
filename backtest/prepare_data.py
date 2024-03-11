import pandas as pd
import numpy as np

class PrepareData(object):
    def __init__(self,
                 factor:pd.DataFrame):
        self.factor = factor
