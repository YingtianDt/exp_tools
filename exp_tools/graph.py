import os
import sys
import glob
import importlib
import _pickle as pickle
from collections import OrderedDict
from .logger import Logger
from .config import Version, Configuration, PolyConfiguration

logger = Logger("GRAPH")

IDENTIFIER_SYMBOL = "@"

def dict_append(d1, d2):
    for k, v in d2.items():
        if k in d1:
            d1[k] = (*d1[k], v)
        else:
            d1[k] = (v, )

def get_id(name, version):
    return f"{name}{IDENTIFIER_SYMBOL}{version}"


def split_id(id):
    tmp = id.split(IDENTIFIER_SYMBOL)
    name = tmp[0]
    version_str = IDENTIFIER_SYMBOL.join(tmp[1:])
    return name, version_str


def split_repr(id):
    name, version_str = split_id(id)
    return f"[{name}] @ {version_str}"

# https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e
# Pickle a file and then compress it into a file with extension


def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'wb') as f:
        pickle.dump(data, f)

# Load any compressed pickle file


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data


class Variable:
    class Default:
        def SAVER(val, stempath):
            with open(stempath+".p", "wb") as f:
                pickle.dump(val, f)

        def LOADER(stempath):
            with open(stempath+".p", "rb") as f:
                data = pickle.load(f)
            return data

    # variable cache with version control
    def __init__(self, name, cache=False, root="./", version=None, saver=None, loader=None, parents=None, children=None):
        self.name = name
        self.version = version if version else Version()
        self.cache = cache
        self.saver = saver  # set return None to avoid pickle cache
        self.loader = loader  # set return None to avoid pickle cache
        self.root = root
        self.parents = parents if parents else set()
        self.children = children if children else set()

    def copy(self):
        ret = Variable(
            name=self.name,
            cache=self.cache,
            root=self.root,
            saver=self.saver,
            loader=self.loader,
            version=self.version.copy(),
            parents=self.parents.copy(),
            children=self.children.copy()
        )
        return ret

    @property
    def identifier(self):
        return get_id(self.name, self.version)

    @property
    def filepath(self):
        ret = [fpath for fpath in glob.glob(self.stempath+".*") if "." not in fpath.replace(self.stempath+".","")]
        assert len(
            ret) <= 1, f"Each version should have 0 or 1 file. But see {ret}."
        if ret:
            return ret[0]
        else:
            return None

    @property
    def stempath(self):
        return os.path.join(self.dirpath, str(self.version))

    @property
    def dirpath(self):
        return os.path.join(self.root, self.name)

    @property
    def cached(self):
        return self.filepath is not None

    @property
    def should_store(self):
        return self.cache and not self.cached

    @property
    def should_remove(self):
        return not self.cache and self.cached

    @property
    def inconsistent(self):
        return self.should_remove or self.should_store

    @property
    def consistent(self):
        return self.cache and self.cached

    @property
    def irrelevant(self):
        return not self.cache and not self.cached

    def remove(self):
        os.remove(self.filepath)
        logger.log(f"Stale variable {split_repr(self.identifier)} removed.")
        for var in self.children:
            if var.cached:
                var.remove()

    def store(self, val):
        os.makedirs(self.dirpath, exist_ok=True)
        # store file
        saver = self.saver if self.saver else self.Default.SAVER
        # e.g. def SAVE_var(val, stempath): save(val, stempath.png)
        val = saver(val, self.stempath)
        logger.log(f"Variable {split_repr(self.identifier)} saved.")

    def load(self):
        try:
            loader = self.loader if self.loader else self.Default.LOADER
            data = loader(self.stempath)
        except Exception as err:
            logger.err(f"Path {self.stempath} fails to be loaded.")
            raise err

        logger.log(f"Variable {split_repr(self.identifier)} loaded.")
        return {self.name: {str(self.version): data}}

    def __repr__(self):
        return self.identifier


def versioned_dict_update(x, y):
    for yvar, yversion in y.items():
        if yvar not in x:
            x[yvar] = yversion
        else:
            xversion = x[yvar]
            xversion.update(yversion)


def load_variables(variables, state_dict):
    for var in variables:
        versioned_dict_update(state_dict, var.load())


def process_parenthesis_statement(s):
    # ignore type hint
    start = s.index("(")
    end = s.rindex(")")
    s = s[start+1:end]
    variables = [w.strip() for w in s.split(",")]
    variables = [w.split(":")[0] if ":" in w else w for w in variables]
    variables = [w for w in variables if w]
    return tuple(variables)


def fetch_lines(filepath: str):
    with open(filepath, "r") as f:
        lines = f.readlines()
        lines = [l[:-1] if l[-1] == '\n' else l for l in lines]
    return lines


class Node:
    @classmethod
    def from_script(self, fpath):
        fname = os.path.basename(fpath)
        lines = fetch_lines(fpath)

        # Node Properties
        info = {
            "name": fname.split('.')[0],
            "context": (os.path.abspath(os.path.join(fpath, os.path.pardir)), ),
            "imports": [],
            "exports": [],
            "cache": tuple(),
        }

        import_onset = False
        export_onset = False
        export_valid = False
        export_last_line = -10
        for i, line in enumerate(lines):

            # parse CACHE
            if "# CACHE:" in line:
                cache = line.replace("# CACHE:", "").strip()
                if cache == "EXPORT":
                    cache = "EXPORT"
                else:
                    cache = (c.strip() for c in cache.split(","))
                info["cache"] = cache

            # parse CONTEXT
            if "# CONTEXT:" in line:
                added_context = (os.path.abspath(c.strip())
                                 for c in line.replace("# CONTEXT:", "").split(","))
                info["context"] = (*context, *added_context)

            # parse REDUCE
            if "# REDUCE" in line:
                info["reduce"] = True

            # parse IMPORT
            if "def SCRIPT" in line:
                import_onset = True

            if "(" in line and import_onset:
                info['imports'].append(i)

            if ")" in line and import_onset:
                info['imports'].append(i)
                import_onset = False

            # alert
            if "# TODO:" in line:
                todo = line.split("# TODO:")[-1].strip()
                logger.warn(f"TODO in {fname}: {todo}")

        for i, line in enumerate(lines[::-1]):

            # parse EXPORT
            if ")" in line and not export_onset:
                export_onset = True
                info['exports'].append(len(lines)-i-1)

            if "(" in line and export_onset:
                info['exports'].append(len(lines)-i-1)
                export_last_line = i

            if "return" in line and export_onset:
                export_onset = False
                export_valid = export_last_line == i or (
                    export_last_line+1) == i
                if export_valid:
                    last_lines = line + lines[len(lines)-export_last_line-1] if export_last_line != i else line
                    tmp = ''.join(last_lines.split("return")[-1])
                    tmp = ''.join(tmp.split("(")[0])
                    tmp = tmp.replace("\\", "")
                    export_valid = tmp.strip() == ""
                break

        if len(info["imports"]) < 2:
            info['imports'] = tuple()
        else:
            imports = info['imports']
            imports = "".join(lines[imports[0]:imports[1]+1])
            info['imports'] = process_parenthesis_statement(imports)

        if len(info["exports"]) < 2 or not export_valid:
            info['exports'] = tuple()
        else:
            exports = info['exports']
            exports = "".join(lines[exports[1]:exports[0]+1])
            info['exports'] = process_parenthesis_statement(exports)

        if info["cache"] == "EXPORT":
            info["cache"] = info['exports']

        return self(**info)

    def __init__(self, name, imports=(), exports=(), context=(), cache=[], flags={}, config=None, root="./", reduce=False):
        self.name = name
        self.root = root
        self.reduce = reduce
        self.imports = imports
        # the dummy variable will not count towards EXPORT caching
        self.exports = exports if exports else (f"{self.name}_EXPORT",)
        self.context = context
        self.export_vars = None
        self.import_vars = None
        self.cache = cache
        self.flags = flags
        self.config = config if config else Configuration()
        self.version = Version()  # only useful when in graph
        self.set_config(self.config)

    def set_config(self, config):
        self.version.update(config)
        self.config = config

    @property
    def stempath(self):
        os.makedirs(self.dirpath, exist_ok=True)  # generate when used
        return os.path.join(self.root, "results", self.identifier)

    @property
    def stemdir(self):
        os.makedirs(self.stempath, exist_ok=True)  # generate when used
        return self.stempath

    @property
    def sharedir(self):
        shardir = os.path.join(self.dirpath, self.name)
        os.makedirs(shardir, exist_ok=True)  # generate when used
        return shardir

    @property
    def dirpath(self):
        return os.path.join(self.root, "results")

    def copy(self):
        ret = Node(
            name=self.name,
            context=self.context,
            cache=self.cache,
            imports=self.imports,
            exports=self.exports,
            flags=self.flags.copy(),
            root=self.root,
            reduce=self.reduce
        )
        if self.export_vars:
            ret.export_vars = {k: v.copy() for k, v in self.export_vars}
        if self.import_vars:
            ret.import_vars = {k: v.copy() for k, v in self.import_vars}
        return ret

    @property
    def poly(self):
        return isinstance(self.config, )

    def __repr__(self):
        return self.identifier

    @property
    def identifier(self):
        return get_id(self.name, self.version)

    @property
    def description(self):
        return "("+",".join(tuple(self.imports))+" -> "+",".join(tuple(self.exports))+")"

    def construct_export_vars(self):
        self.export_vars = OrderedDict()
        for exp in self.exports:
            var = Variable(exp, cache=exp in self.cache,
                           root=os.path.join(self.root, "cache"))
            var.version += self.version  # inherit configs
            self.export_vars[var.identifier] = var
        return self.export_vars

    def set_import_vars(self, import_vars: OrderedDict):
        self.import_vars = import_vars

    def get_var(self, identifier):
        if var in self.export_vars:
            return self.export_vars[identifier]
        else:
            return self.import_vars[identifier]

    def fetch_inputs(self, state_dict):
        inputs = {}  # {varname: val}
        for var in self.import_vars.values():
            value = state_dict[var.name][str(var.version)]
            if self.reduce:
                if var.name not in inputs:
                    inputs[var.name] = {str(var.version): value}
                else:
                    inputs[var.name].update({str(var.version): value})
            else:
                inputs[var.name] = value

        if self.reduce:
            return {varname: self.xarrary_compress(var_dict) for varname, var_dict in inputs.items()} 
        else:
            return inputs

    def format_outputs(self, ret):
        # return {varname: {version: val}}
        return {var.name: {str(var.version): r} for var, r in zip(self.export_vars.values(), ret)}

    # helper for compress inputs from different versions into xarray
    def xarrary_compress(self, var_dict):
        # var_dict: {version1: val1, version2: val2, ...} or {default: v}
        assert self.reduce
        if "default" in var_dict: return var_dict["default"]
        
        import xarray as xr
        import numpy as np

        versions = OrderedDict()
        for version_str, value in var_dict.items():
            version = Version.from_str(version_str)
            dict_append(versions, version)

        coords = OrderedDict()
        for k,v in versions.items():
            coords.setdefault(k, set()).update(v)
        for k,v in versions.items():
            tmp = list(coords[k])
            tmp.sort()
            coords[k] = tmp
        coord_lists = [[{k:v} for v in vs] for k,vs in coords.items()]

        # https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
        def cartesian_product(*arrays):
            la = len(arrays)
            arr = np.empty([len(a) for a in arrays] + [la], dtype=np.object)
            for i, a in enumerate(np.ix_(*arrays)):
                arr[...,i] = a
            return arr

        index = cartesian_product(*coord_lists)
        index = np.apply_along_axis(Version.from_dicts, axis=-1, arr=index)
        data = np.vectorize(lambda x: var_dict[str(x)], otypes=[np.object])(index)
        xarr = xr.DataArray(data, coords=coords, dims=coords.keys())
        return xarr

    def load(self):
        # load script, module, logger, view, config
        self.module = __import__(self.name)
        del sys.modules[self.name]  # TODO:force to reimport everything

        if hasattr(self.module, "SCRIPT"):
            self.script = self.module.SCRIPT
        else:
            logger.fatal(f"File {self.name}.py does not have a SCRIPT.")

        if hasattr(self.module, "logger"):
            if not self.module.logger.name:
                logger_name = self.identifier.upper()
                self.module.logger = Logger(logger_name)

        if hasattr(self.module, "node"):
            self.module.node = self

        if self.config:
            if hasattr(self.module, "config"):
                self.module.config = self.config
            else:
                logger.fatal(
                    f"Node {self.name}.py is specified by configurations, but does not use it.")

        if hasattr(self.module, "config") and not self.config:
            logger.fatal(f"Node {self.name}.py requires configurations.")

        # local = dir(self.module)
        # for var in self.exports:
        #     saver_func = f"SAVE_{var}"
        #     if saver_func in local:
        #         var.saver = getattr(self.module, saver_func)
        #     loader_func = f"LOAD_{var}"
        #     if loader_func in local:
        #         var.loader = getattr(self.module, loader_func)


def varname_match_id(name, identifier):
    return name == identifier.split(IDENTIFIER_SYMBOL)[0]


def get_graph(nodes):
    graph = {node: set() for node in nodes}
    exports = set()
    for node in nodes:
        for other in nodes:
            if set(node.exports).intersection(set(other.exports)):
                if other is not node:
                    logger.fatal(f"Nodes [{node.name}.py] and [{other.name}.py] have common exports.")

            if set(node.exports).intersection(set(other.imports)):
                if other is node:
                    logger.fatal(f"Node [{node.name}.py] is cyclic.")
                graph[node].add(other)
    return graph


def find_descendents(node, graph):
    stack = set()

    def find_descendents_helper(node):
        ret = set()
        stack.add(node)
        for child in graph[node]:
            if child in stack:
                logger.fatal(f"Loop detected at {child.name}.py.")
            else:
                ret.add(child)
                ret.update(find_descendents_helper(child))
        stack.remove(node)
        return ret

    return find_descendents_helper(node)


def init_node_topogoly(nodes, node_configs):
    graph = get_graph(nodes)
    node_graphconfig = {node: PolyConfiguration() for node in nodes}  # version propagate
    node_config_keys = {node: node_configs.get(node.name, {}).keys() for node in nodes}  # what node gets configured
    nodes = {node.name: node for node in nodes}

    # check config
    for node_name in node_configs:
        if node_name not in nodes:
            logger.fatal(f"[{node_name}] configured, but the node does not exist.")

    # propagate polyconfigs
    for node_name, polyconfig in node_configs.items():
        if node_name in nodes:
            node = nodes[node_name]
            node_graphconfig[node] += polyconfig
            descendents = find_descendents(node, graph)
            for desc in descendents:
                node_graphconfig[desc] += polyconfig

    # factor base graph
    factor_nodes = []
    for node, polyconfig in node_graphconfig.items():
        if node.reduce:
            node_config = {k: polyconfig[k]
                        for k in node_config_keys[node]}
            node.set_config(Configuration.from_dict(node_config))
            node.version.add(polyconfig)
            factor_nodes.append(node)
        else:
            graphconfigs = polyconfig.factor()
            for graphconfig in graphconfigs:
                new_node = node.copy()
                new_node_config = {k: graphconfig[k]
                                for k in node_config_keys[node]}
                new_node.set_config(Configuration.from_dict(new_node_config))
                new_node.version.add(graphconfig)
                factor_nodes.append(new_node)

    # comstruct variables and topology
    var2node = {}
    var2node_ = {}
    variables = {}
    for node in factor_nodes:
        export_vars = node.construct_export_vars()
        for varid, var in export_vars.items():
            var2node[var] = node  # a variable is produced by only one node
            variables[varid] = var
            if var.name in var2node_:
                var2node_[var.name][var.version] = node
            else:
                var2node_[var.name] = {var.version: node}

    def version_contain(x, y):
        for k,v in y.items():
            if k in x and x[k] != y[k]:
                return False
        return True

    def find_import_version(import_varname, node):
        export_version = node.version
        if import_varname not in var2node_:
            logger.fatal(f"Variable {import_varname} required by [{node.name}.py] not generated by any node.")
        for version in var2node_[import_varname]:
            if version_contain(export_version, version):
                return version
        logger.fatal("Version not found.")

    # link import variables
    for node in factor_nodes:
        import_vars = OrderedDict()
        for varname in node.imports:
            if node.reduce:
                for varid in variables:
                    if varname == split_id(varid)[0]:
                        import_vars[varid] = variables[varid]
            else:
                version = find_import_version(varname, node)
                varid = get_id(varname, version)
                import_vars[varid] = variables[varid]

        node.set_import_vars(import_vars)
        export_vars = node.export_vars

        # construct topology
        for import_var in import_vars.values():
            import_var.children.update(export_vars.values())
        for export_var in export_vars.values():
            export_var.parents.update(import_vars.values())

    nodes = {node.identifier: node for node in factor_nodes}
    return nodes, variables, var2node

def ticker(t, list):
    # t = {k:v}
    for e in list:
        if isinstance(e, Version):
            e.add(t)
        else:
            ticker(t, e)
    return list

def xarr_helper(coords):
    # coords = [k:[v1,v2,...], ...]
    if not coords: return tuple()
    coord = coords[0]
    remain = coords[1:]
    k = list(coord.keys())[0]
    vs = coord[k]
    if not remain:
        return tuple((Version({k:v}) for v in vs))

    ret = []
    for v in vs:
        l = xarr_helper(remain)
        ret.append(ticker({k:v}, l))
    return tuple(ret)

def construct_xarr(coords):
    coords = [{k:v} for k,v in coords.items()]
    return xarr_helper(coords)

def fetch_vals(index):
    if not index: return tuple
    if isinstance(index[0], Version):
        return tuple((var_dict[str(v)] for v in index))
    else:
        return tuple((fetch_vals(i) for i in index))

node = Node("")
