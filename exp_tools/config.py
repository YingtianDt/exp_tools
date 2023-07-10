FORBIDDEN = "/\\?`:<>|*\"\'"
DEFAULT_VERSION = "default"
CONFIG_SEPARATOR = "_"
CONFIG_KV_SEPARATOR = "."
assert CONFIG_SEPARATOR not in FORBIDDEN
assert CONFIG_KV_SEPARATOR not in FORBIDDEN

from functools import reduce

class Configuration(dict):
    VAL_TYPES = (int, float, bool, str)
    FORBIDDEN_CHAR = FORBIDDEN + CONFIG_SEPARATOR
    # configuration: set of k->v, whose print is ordered by alphabet order
    def _check_val(self, v):
        if isinstance(v, tuple):
            for i in v:
                self._check_val(i)
        else:
            v_str = str(v)
            for c in self.FORBIDDEN_CHAR:
                assert c not in v_str, c
            assert type(v) in self.VAL_TYPES

    def _check_key(self, k):
        assert isinstance(k, str)
        assert k.isalpha()
        for c in self.FORBIDDEN_CHAR:
            assert c not in k

    def __init__(self, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        for k, v in self.items():
            self._check_kv(k, v)

    def __setitem__(self, k, v):
        if k in self:
            raise KeyError(f"Key {k} already exists.")
        self._check_kv(k, v)
        super(Configuration, self).__setitem__(k, v)

    def _check_kv(self, k, v):
        try:
            self._check_key(k), 
        except:
            raise KeyError(f"Type of key {k} not accepted.")
        try:
            self._check_val(v)
        except:
            raise ValueError(f"Type of value {v} not accepted.")

    @staticmethod
    def from_dict(dict):
        c = Configuration()
        for k,v in dict.items():
            c[k] = v
        return c

    def __add__(self, other):
        ret = self.copy()
        ret.update(other)
        return ret

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def copy(self):
        return (type(self))(super(Configuration, self).copy())

class PolyConfiguration(Configuration):
    # Eg. {a:[1,2], b:3}
    def _check_val(self, v):
        if isinstance(v, tuple) or isinstance(v, list):
            for i in v:
                self._check_val(i) 
        else:
            v_str = str(v)
            for c in self.FORBIDDEN_CHAR:
                assert c not in v_str, c
            assert type(v) in self.VAL_TYPES

    def _is_poly_val(self, v):
        return isinstance(v, list)  # use list for poly config, use tuple for config a iterable

    def factor(self):
        configs = [Configuration()]
        for k,v in self.items():
            if self._is_poly_val(v):
                N = len(v)
                new_configs = []
                for config in configs:
                    for n in range(N):
                        new_config = config.copy()
                        new_config[k] = v[n]
                        new_configs.append(new_config)
                configs = new_configs
            else:
                for config in configs:
                    config[k] = v
        return configs

    @staticmethod
    def from_dict(d):
        assert isinstance(d, dict)
        c = PolyConfiguration()
        for k,v in d.items():
            c[k] = v
        return c

# class Version(list):
#     # Representation of ordered configurations
#     def add(self, config):
#         # same level
#         self[-1].update(config)

#     def append(self, config):
#         # append
#         assert isinstance(config, Configuration)
#         self.append(config)

#     def __repr__(self):
#         if len(self) == 0:
#             return "default"
#         return "#".join([str(c) for c in self])

#     def __add__(self, x):
#         for i, (a, b) in enumerate(zip(self, x)):
#             self[i] = a+b

class Version(dict):
    # Representation of ordered configurations
    def add(self, config):
        self.update(config)

    def copy(self):
        return (type(self))(super(Version, self).copy())

    def __add__(self, other):
        ret = self.copy()
        ret.update(other)
        return ret

    def __sub__(self, other):
        ret = self.copy()
        for k,v in other.items():
            assert ret[k] == v
            del ret[k]
        return ret

    def __repr__(self):
        s = []
        for k, v in sorted(self.items()):
            s.append(f"{k}{CONFIG_KV_SEPARATOR}{str(v).replace(' ', '')}") 
        if len(s) == 0:
            return DEFAULT_VERSION
        else:
            s = CONFIG_SEPARATOR.join(s)
            return s

    def isdefault(self):
        return len(self) == 0

    @staticmethod
    def from_dict(dict):
        c = Version()
        for k,v in dict.items():
            c[k] = v
        return c

    @staticmethod
    def from_dicts(dicts):
        c = reduce(lambda a,b:a+b, (Version.from_dict(d) for d in dicts))
        return c
    
    @staticmethod
    def from_str(string):
        configs = string.split(CONFIG_SEPARATOR)
        ret = {}
        for c in configs:
            tmp = c.split(CONFIG_KV_SEPARATOR)
            string = '.'.join(tmp[1:])
            try:
                val = eval(string)
            except NameError:
                val = string
            ret[tmp[0]] = val
        return Version(ret)

    def __hash__(self):
        return hash(str(self))

config = Configuration()
version = Version()