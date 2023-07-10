import os
import pickle
from collections import OrderedDict
from .logger import Logger

logger = Logger("CACHE")

class Cache:
    def __init__(self, variables, path="./cache"):
        self.variables = variables
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.state = {var:False for var in variables}

        # check existing cache
        for fname in os.listdir(self.path):
            var = fname.split(".")[0]
            if var in self.state:
                self.state[var] = True
            else:
                logger.log(f"Stale variable {var} removed.")
                os.remove(self.get_path(var))
    
    def contain(self, var):
        return var in self.variables

    def iscached(self, var):
        return self.contain(var) and self.state[var]

    def get_path(self, var):
        return os.path.join(self.path, var+".p")

    def dump(self, var, val):
        with open(self.get_path(var), "wb") as f:
            pickle.dump(val, f)
        logger.log(f"Variable {var} dumped.")

    def load(self, var):
        with open(self.get_path(var), "rb") as f:
            try:
                data = pickle.load(f)
            except Exception as err:
                logger.err(f"Path {self.get_path(var)} fails to be loaded.")
                raise err

        logger.log(f"Variable {var} loaded.")
        return {var: data}

    def dumps(self, vals):
        for var, val in vals.items():
            self.dump(var, val)
    
    def loads(self, vars):
        ret = {}
        for var in vars:
            ret.update(self.load(var))
        return ret

    def loads_state(self, vars, state_dict):
        state_dict.update(self.loads(vars))

    def load_state(self, var, state_dict):
        state_dict.update(self.load(var))






