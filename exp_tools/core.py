import os
import glob
import sys
from collections import OrderedDict
from .logger import Logger
from .cache import Cache
from . import probe
from .probe import timeit
from .config import Configuration, PolyConfiguration, Version
from .graph import Node, Variable, init_node_topogoly, split_id, load_variables, versioned_dict_update, IDENTIFIER_SYMBOL

logger = Logger("CORE")


def topological_sort(graph):

    is_visit = OrderedDict((node, False) for node in graph)
    li = []

    def dfs(graph, start_node):

        if start_node in start_node.children:
            logger.fatal(f"Error: Self loop detected for {start_node}.")

        dfs_visit[start_node] = True
        dfs_stack.add(start_node)

        for end_node in start_node.children:
            if not dfs_prohibit[end_node]:
                if not dfs_visit[end_node]:
                    dfs_visit[end_node] = True
                else:
                    if end_node in dfs_stack:
                        backedge = f"{end_node.name} -> {start_node.name}"
                        logger.fatal(f"Error: Loop detected among nodes.", "\nBack edge:", backedge)

        for end_node in start_node.children:
            if not is_visit[end_node]:
                is_visit[end_node] = True
                dfs(graph, end_node)

        li.append(start_node)
        dfs_stack.remove(start_node)

    for start_node in graph:
        if not is_visit[start_node]:
            dfs_prohibit = OrderedDict((node, is_visit[node]) for node in is_visit)
            is_visit[start_node] = True
            dfs_visit = OrderedDict((node, False) for node in graph)
            dfs_visit[start_node] = True
            dfs_stack = set()
            dfs(graph, start_node)

    li.reverse()
    return li


class Executor:
    def __init__(self, nodes, node_configs={}, global_configs={}, flags={}, output_dir="./"):
        timeit(start=True)
        self.node_configs = {k:PolyConfiguration.from_dict(v) for k,v in node_configs.items()}
        self.global_configs = PolyConfiguration.from_dict(global_configs)
        self.flags = flags
        self.output_dir = output_dir
        probe.set_store_dir(os.path.join(output_dir, "probe"))

        # load flags
        self.debug = self.flags.get("debug", False)
        self.load_cache = self.flags.get("load_cache", True)
        self.store_cache = self.flags.get("store_cache", True)
        self.verify_cache = self.flags.get("verify_cache", True)
        self.version = Version.from_dict(self.flags.get("version", {}))

        for node in nodes: 
            node.root = self.output_dir
            node.version += self.version

        self.nodes, self.variables, self.var2node = init_node_topogoly(nodes, self.node_configs)
        self.topology = topological_sort(self.variables.values())  # execution sequence

    def visualize(self):
        import networkx as nx
        from matplotlib import pyplot as plt

        dag = OrderedDict({str(v):[str(c) for c in v.children] for v in self.variables.values()})

        def id_format(id):
            name, version = split_id(id)
            return f"{name}\n@\n{version}"

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
        }


        plt.subplots(figsize=(8,8))
        nx.draw_networkx_nodes(g, pos=pos, **node_style)
        # nx.draw_networkx_edges(g, pos=pos, **edge_style)
        nx.draw_networkx_labels(g, pos=pos, labels={k:id_format(k) for k in dag.keys()}, **label_style)

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
        parents = var.parents
        ascendents.update(parents)
        for p in parents:
            ascendents.update(self.find_ascendents(p))
        return ascendents

    def find_necessary_ascendents(self, var):
        ascendents = set()
        if var.consistent:
            return ascendents
        parents = var.parents
        ascendents.update(parents)
        for p in parents:
            ascendents.update(self.find_necessary_ascendents(p))
        return ascendents


    def find_descendents(self, var):
        descendents = set()
        children = var.children
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

    def find_nodes(self, node_name):
        try:
            ret = set()
            if node_name == "*":
                ret.update(self.nodes)
            elif node_name.startswith("*"):
                endswith = node_name.strip("*")
                for node_id, node in self.nodes.items():
                    name, _ = split_id(node_id)
                    if name.endswith(endswith): ret.add(node)
            elif node_name.endswith("*"):
                startswith = node_name.strip("*")
                for node_id, node in self.nodes.items():
                    name, _ = split_id(node_id)
                    if name.startswith(startswith): ret.add(node)
            else:
                for node_id, node in self.nodes.items():
                    name, _ = split_id(node_id)
                    if name == node_name:
                        ret.add(node)
            return ret 
        except KeyError:
            raise logger.fatal(f"Node [{node_name}.py] not found or inactive.")

    def exec_single_node(self, node, state_dict):
        loc_state = node.fetch_inputs(state_dict)
        while True:
            logger.log(f"Executing [{node.name}.py] @ {node.version} ...")
            try:
                node.load()  # reload to ensure new code run
            except Exception as err:
                logger.log(f"[{node.name}.py] failed to load.")
                raise err
            try:
                ret = node.script(**loc_state)
                break
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
                    ret = input("[OPTION]:")
                    if ret == "q": exit()
                else:
                    exit()

        ret = (ret,) if not isinstance(ret, tuple) else ret
        return_state = node.format_outputs(ret)
        versioned_dict_update(state_dict, return_state)
        return state_dict

    # core function
    def exec_variables(self, variables, state_dict):
        # compute dependent variables
        vars = set(variables)
        var_cache = {v: v.cache for v in vars}
        dependency = set(vars)
        for var in vars: var.cache = False  # force execution
        for var in vars:
            if self.load_cache:
                dependency.update(self.find_necessary_ascendents(var))
            else:
                dependency.update(self.find_ascendents(var))

        for var in vars: var.cache = var_cache[var]  # restore cache assignment
        timeit("Compute Dependency")

        # load cached variables
        if self.verify_cache or self.load_cache:

            # verify cache
            if self.verify_cache:
                for var in dependency:
                    if var.should_remove:
                        var.remove()

            # load cache
            if self.load_cache:
                cached = set()

                # after the variable removal
                for var in dependency:
                    if var.consistent and var not in vars:
                        cached.add(var)

                dependency = dependency - cached
                load_variables(cached, state_dict)

        timeit("Load Cache")

        # find necessary nodes and execute
        dependency = list(dependency)
        dependency.sort(key=lambda v: self.topology.index(v))
        nodes = self.cover_dependencies(dependency)

        # prepare necessary context
        for node in nodes:
            for c in node.context:
                if c not in sys.path:
                    sys.path.append(c)

        # # load modules:
        # for node in nodes:
        #     node.load()

        timeit("Load Nodes")

        # execute in order
        for n in nodes:
            self.exec_single_node(n, state_dict)
            # store cache
            if self.store_cache:
                for v in n.export_vars.values():
                    if v.should_store or v.cached:  # v.cached also works because we just want to get the newest version
                        v.store(state_dict[v.name][str(v.version)])

        logger.log("Execution finished.")
        return state_dict

    def exec_nodes(self, node_names, state_dict):
        nodes = set()
        for name in node_names:
            nodes.update(self.find_nodes(name)) 
        exports = set()
        for node in nodes:
            exports.update(node.export_vars.values())
        return self.exec_variables(exports, state_dict)

    def exec_node(self, node_name, state_dict):
        return self.exec_nodes([node_name], state_dict)

