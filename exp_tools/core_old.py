import os
import shutil 
import sys
import pickle
from collections import OrderedDict
from .logger import Logger
from .cache import Cache
from . import probe
from .probe import timeit
from .config import Configuration

logger = Logger("CORE")


class Variable:
    # variable cache with version control
    def __init__(self, name, cache=False, root="./", config=None, saver=None, loader=None):
        self.name = name
        self.config = config if config else Configuration()
        self.cache = cache
        self.saver = saver  # set return None to avoid pickle cache
        self.loader = loader  # set return None to avoid pickle cache
        self.root = root

    @property
    def filepath(self):
        return self.stempath + ".p"

    @property
    def stempath(self):
        return os.path.join(self.dirpath, str(self.config))

    @property
    def dirpath(self):
        return os.path.join(self.root, self.name)

    @property
    def cached(self):
        return os.path.exists(self.filepath)

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
        shutil.rmtree(self.dirpath)
        logger.log(f"Stale variable {self.name} @ {self.config} removed.")

    def store(self, val):
        os.makedirs(self.dirpath, exist_ok=True)

        # check if view and run
        if self.saver:
            val = self.saver(val, self.stempath)  # e.g. def SAVE_var(val, stempath): save(val, stempath.png)
            logger.log(f"Variable {self.name} @ {self.config} saved.")

        # store file
        if val:
            with open(self.filepath, "wb") as f:
                pickle.dump(val, f)
            logger.log(f"Variable {self.name} @ {self.config} pickle-dumped.")
        else:
            self.cache = False  # avoid be counted as a dependency

    def load(self):
        try:
            if self.loader:
                data = self.loader(self.stempath)
            else:
                with open(self.filepath, "rb") as f:
                    data = pickle.load(f)
        except Exception as err:
            logger.err(f"Path {self.filepath} fails to be loaded.")
            raise err

        logger.log(f"Variable {self.name} loaded.")
        return {self.name: data}

    def __repr__(self):
        return f"[{self.name}]"


def load_variables(variables, state_dict):
    for var in variables:
        state_dict.update(var.load())


def process_parenthesis_statement(s):
    # ignore type hint
    start = s.index("(")
    end = s.rindex(")")
    s = s[start+1:end]
    variables = [w.strip() for w in s.split(",")]
    variables = [w.split(":")[0] if ":" in w else w for w in variables ]
    variables = [w for w in variables if w]
    return variables

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
            "identifier": fname.split('.')[0],
            "context": (os.path.abspath(os.path.join(fpath, os.path.pardir)), ),
            "imports": [],
            "exports": [],
            "cache": tuple()
        }

        import_onset = False
        export_onset = False
        export_valid = False
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
                added_context = (os.path.abspath(c.strip()) for c in line.replace("# CONTEXT:", "").split(","))
                info["context"] = (*context, *added_context)

            # parse IMPORT
            if "def SCRIPT" in line:
                import_onset = True

            if "(" in line and import_onset:
                info['imports'].append(i)

            if ")" in line and import_onset:
                info['imports'].append(i)
                import_onset = False
                break

            # alert
            if "# TODO:" in line:
                todo = line.replace("# TODO:", "").strip()
                logger.warn(f"TODO in {fname}: {todo}")

        for i, line in enumerate(lines[::-1]):

            # parse EXPORT
            if ")" in line:
                export_onset = True
                info['exports'].append(len(lines)-i-1)

            if "(" in line and export_onset:
                info['exports'].append(len(lines)-i-1)

            if "return" in line and export_onset:
                export_onset = False
                export_valid = True
                break

        if len(info["imports"]) < 2:
            info['imports'] = []
        else:
            imports = info['imports']
            imports = "".join(lines[imports[0]:imports[1]+1])
            info['imports'] = process_parenthesis_statement(imports)

        if len(info["exports"]) < 2 or not export_valid:
            info['exports'] = []     
        else:
            exports = info['exports']
            exports = "".join(lines[exports[1]:exports[0]+1])
            info['exports'] = process_parenthesis_statement(exports)

        if info["cache"] == "EXPORT":
            info["cache"] = info['exports'] 

        return self(**info)

    def __init__(self, identifier, imports, exports, context=(), cache=[], flags={}):
        self.identifier = identifier
        self.imports = imports
        # the dummy variable will not count towards EXPORT caching
        self.exports = exports if exports else (f"{self.identifier}_EXPORT",)  
        self.context = context
        self.cache = cache
        self.flags = flags

    def __repr__(self):
        return self.identifier

    @property
    def description(self):
        return "("+",".join(tuple(self.imports.keys()))+" -> "+",".join(tuple(self.exports.keys()))+")"

    def load(self, config=None):
        # load script, module, logger, view, config
        self.module = __import__(self.identifier)
        if hasattr(self.module, "SCRIPT"):
            self.script = self.module.SCRIPT 
        else:
            logger.fatal(f"File {self.identifier} does not have a SCRIPT.")

        if hasattr(self.module, "logger"):
            if not self.module.logger.name:
                logger_name = self.identifier.upper() if self.stage==self.identifier else f"{self.stage.upper()}:{self.identifier}"
                self.module.logger.set_name(logger_name)

        if config:
            if hasattr(self.module, "config"):
                self.module.config = config
            else:
                logger.fatal(f"Node {self.identifier}.py is specified by configurations, but does not use it.")

        if hasattr(self.module, "config") and not config:
            logger.fatal(f"Node {self.identifier}.py requires configurations.")

        local = dir(self.module)
        for var in self.exports:
            saver_func = f"SAVE_{var.name}"
            if saver_func in local:
                var.saver = getattr(self.module, saver_func)
            loader_func = f"LOAD_{var.name}"
            if loader_func in local:
                var.loader = getattr(self.module, loader_func)


def dict_combine(d1, d2):
    for k, v in d2.items():
        if k in d1:
            d1[k] = (*d1[k], *v)
        else:
            d1[k] = (*v, )

def construct_helper(nodes):
    variables = set()
    cache = set()
    for node in nodes:
        variables.update(node.imports)
        variables.update(node.exports)

    var2node = {}
    var2parent = {v:set() for v in variables}
    var2children = {v:set() for v in variables}
    for node in nodes:
        for exp in node.exports:
            if exp in node.cache:
                cache.add(exp)
            var2node[exp] = node
            var2parent.setdefault(exp, set()).update(node.imports)
        for imp in node.imports:
            var2children.setdefault(imp, set()).update(node.exports)
    return variables, var2node, var2parent, var2children, cache


def topological_sort(graph):

    is_visit = OrderedDict((node, False) for node in graph)
    li = []

    def dfs(graph, start_node):

        if start_node in graph[start_node]:
            logger.fatal(f"Error: Self loop detected for {start_node}.")

        dfs_visit[start_node] = True
        dfs_stack.add(start_node)

        for end_node in graph[start_node]:
            if not dfs_prohibit[end_node]:
                if not dfs_visit[end_node]:
                    dfs_visit[end_node] = True
                else:
                    if end_node in dfs_stack:
                        # graph_str = "\n".join([f"{k}\t->\t{v}" for k,v in graph.items()])
                        backedge = f"{end_node} -> {start_node}"
                        logger.fatal(f"Error: Loop detected among nodes.", "\nBack edge:", backedge)

        for end_node in graph[start_node]:
            if not is_visit[end_node]:
                is_visit[end_node] = True
                dfs(graph, end_node)

        li.append(start_node)
        dfs_stack.remove(start_node)

    for start_node in graph:
        if not is_visit[start_node]:
            dfs_prohibit = is_visit.copy()
            is_visit[start_node] = True
            dfs_visit = OrderedDict((node, False) for node in graph)
            dfs_visit[start_node] = True
            dfs_stack = set()
            dfs(graph, start_node)

    li.reverse()
    return li


class Executor:
    def __init__(self, nodes, node_configurations={}, flags={}, output_dir="./"):
        self.configurations = {k:Configuration.from_dict(v) for k,v in node_configurations.items()}
        self.flags = flags
        self.output_dir = output_dir

        timeit(start=True)

        variables, var2node, var2parent, var2children, cache = construct_helper(nodes)
        self.variables = {v:Variable(v, cache=v in cache) for v in variables}
        self.nodes = {node.identifier:self._link_variables_helper(node) for node in nodes}
        self.var2node = self._link_variables_helper(var2node)
        self.var2children = self._link_variables_helper(var2children)
        self.var2parent = self._link_variables_helper(var2parent)
        self.topology = topological_sort(self.var2children)  # execution sequence

        self.load_configuration()

        # load flags
        self.debug = self.flags.get("debug", False)
        self.version = self.flags.get("version", {})
        for v in self.variables.values(): 
            v.config += self.version
            v.root = os.path.join(self.output_dir, "cache")
        probe.set_store_dir(os.path.join(output_dir, "probe"))

    def load_configuration(self):
        for node_id in self.configurations:
            node = self.nodes[node_id]
            exports = node.exports
            descendents = set(exports)
            for exp in exports:
                descendents.update(self.find_descendents(exp))
                
            for variable in descendents:
                variable.config += self.configurations[node_id]

    def _link_variables_helper(self, data):
        vs = self.variables
        if isinstance(data, dict):
            ret = {}
            for k,v in data.items():
                if isinstance(v, set):
                    ret[vs[k]] = set((vs[l] for l in v))
                else:
                    ret[vs[k]] = v
            return ret
        elif isinstance(data, Node):
            data.imports = [vs[x] for x in data.imports]
            data.exports = [vs[x] for x in data.exports]
            return data

    def visualize(self):
        import networkx as nx
        from matplotlib import pyplot as plt

        dag = {str(k):[str(t) for t in v] for k,v in self.var2children.items()}

        g = nx.DiGraph()
        g.add_nodes_from(dag.keys())
        for k,v in dag.items():
            g.add_edges_from([(k,t) for t in v])
        pos = nx.kamada_kawai_layout(g)
        pos = nx.spring_layout(g, pos=pos, iterations=5)
        
        d = dict(g.degree)

        node_style = {
            'node_color': 'lightblue',
            'node_size': 2000,
            'linewidths': 2,
        }
        
        label_style = {
            'font_size': 8,
            'labels': {k:k for k in dag.keys()}
        }


        plt.subplots(figsize=(8,8))
        nx.draw_networkx_nodes(g, pos=pos, **node_style)
        # nx.draw_networkx_edges(g, pos=pos, **edge_style)
        nx.draw_networkx_labels(g, pos=pos, **label_style)

        ax = plt.gca()
        for e in g.edges:
            r = str(0.3*e[2]) if len(e) == 3 else '0'
            ax.annotate("",
                        xy=pos[e[1]], xycoords='data',
                        xytext=pos[e[0]], textcoords='data',
                        arrowprops=dict(arrowstyle="->", color="0.5",
                                        shrinkA=35, shrinkB=35,
                                        patchA=None, patchB=None, linewidth=3.5,
                                        alpha=0.5,
                                        connectionstyle="arc3,rad=rrr".replace('rrr', r),
                                        ),
                        )
        plt.axis('off')
        plt.margins(x=.2, y=.2)
        plt.show()

    def find_ascendents(self, var):
        ascendents = set()
        if var.consistent:
            return ascendents
        try:
            parents = self.var2parent[var]
        except KeyError:
            raise logger.fatal(f"Variable {var.name} is not generated by any script.")
        ascendents.update(parents)
        for p in parents:
            ascendents.update(self.find_ascendents(p))
        return ascendents

    def find_descendents(self, var):
        descendents = set()
        try:
            children = self.var2children[var]
        except KeyError:
            raise logger.fatal(f"Variable {var.name} is not generated by any script.")
        descendents.update(children)
        for c in children:
            descendents.update(self.find_descendents(c))
        return descendents

    def cover_dependencies(self, vars):
        # the order matters: execution order
        nodes_to_exec = []
        for var in vars:
            node = self.var2node[var]
            if node not in nodes_to_exec:
                nodes_to_exec.append(node)
            vars = [v for v in vars if v not in node.exports]
            if len(vars) == 0: break
        return nodes_to_exec

    def find_nodes(self, node_identifier):
        try:
            ret = set()
            if node_identifier == "*":
                ret.update(self.nodes)
            elif node_identifier.startswith("*"):
                endswith = node_identifier.strip("*")
                for node_name in self.nodes:
                    if node_name.endswith(endswith): ret.add(self.nodes[node_name])
            elif node_identifier.endswith("*"):
                startswith = node_identifier.strip("*")
                for node_name in self.nodes:
                    if node_name.startswith(startswith): ret.add(self.nodes[node_name])
            else:
                ret.add(self.nodes[node_identifier])
            return ret 
        except KeyError:
            raise logger.fatal(f"Node [{node_identifier}.py] not found or inactive.")

    def exec_single_node(self, node, state_dict):
        loc_state = {k.name:state_dict[k.name] for k in node.imports}
        try:
            logger.log(f"Executing [{node.identifier}.py] ...")
            ret = node.script(**loc_state)
        except Exception as err:
            import traceback
            exc_type, exc_value, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            tb_info = tb_info[1:]
            logger.log("Error encountered.")
            logger.log("Traceback:")
            logger.log("".join(traceback.format_list(tb_info)))
            logger.log(err)
            if self.debug:
                logger.log('')
                logger.log("PDB debugger opened...")
                import pdb; pdb.post_mortem(tb)
            exit()

        ret = (ret,) if not isinstance(ret, tuple) else ret
        return_state = {k.name: v for k,v in zip(node.exports, ret)}
        state_dict.update(return_state)
        return state_dict

    # core function
    def exec_variables(self, variables, state_dict):
        # compute dependent variables
        vars = set(variables)
        dependency = set(vars)
        for var in vars:
            dependency.update(self.find_ascendents(var))

        timeit("Compute Dependency")

        # load cached variables
        cached = set()
        for var in dependency:
            if var.consistent:
                cached.add(var)
            if var.should_remove:
                var.remove()
        dependency = dependency - cached
        load_variables(cached, state_dict)

        timeit("Load Cache")

        # find necessary nodes and execute
        dependency = list(dependency)
        dependency.sort(key=lambda v: self.topology.index(v))
        nodes = self.cover_dependencies(dependency)

        # prepare context
        for node in nodes:
            for c in node.context:
                if c not in sys.path:
                    sys.path.append(c)

        # load modules:
        for node in nodes:
            node.load(config=self.configurations.get(node.identifier))

        timeit("Load Nodes")

        for n in nodes:
            self.exec_single_node(n, state_dict)
            for v in n.exports:
                if v in dependency and v.should_store:
                    v.store(state_dict[v.name])

        logger.log("Execution finished.")
        return state_dict

    def exec_nodes(self, node_identifiers, state_dict):
        nodes = set()
        for nid in node_identifiers:
            nodes.update(self.find_nodes(nid)) 
        exports = set()
        for node in nodes:
            exports.update(node.exports)
        return self.exec_variables(exports, state_dict)

    def exec_node(self, node_identifier, state_dict):
        return self.exec_nodes([node_identifier], state_dict)

