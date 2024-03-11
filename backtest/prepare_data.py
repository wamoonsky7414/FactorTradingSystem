import pandas as pd
import numpy as np

class PrepareData(object):
    def __init__(self,
                 factor:str,
                 market:str):
        self.factor = factor
        self.market = market


    def get_factor_df(self):


