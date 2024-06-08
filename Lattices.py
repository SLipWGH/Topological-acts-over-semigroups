from typing import Callable

import networkx as nx
import numpy.typing as npt

from Topology import topological_infimum, topological_supremum


def get_topologies_list_ordered_by_power(
    S: set[frozenset[frozenset[int]]],
)-> list[frozenset[frozenset[int]]]:
    ordered = []
    ordered.extend(S)
    ordered.sort(key=len)
    return ordered


def get_edges_set(
    S: list[frozenset],
    infimum: Callable
)->set:
    edges = []
    max_by_inclusion = []
    for i, A in enumerate(S):
        for j, B in enumerate(S):
            if A != B and infimum(A, B) == B:
                if max_by_inclusion and j:
                    buff = max_by_inclusion.copy()
                    for idx, k in enumerate(max_by_inclusion):
                        if B > S[k]:
                            buff[idx] = j
                        elif not B < S[k] and j not in buff:
                            buff.append(j)
                    max_by_inclusion = buff
                else:
                    max_by_inclusion.append((j))
        for j in max_by_inclusion:
            edges.append((i, j))
        max_by_inclusion.clear()
    return edges


def get_graph_by_topologies(
    vertices: set[frozenset[frozenset[int]]],
)->nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(vertices)))
    G.add_edges_from(get_edges_set(vertices, topological_infimum))
    return G


def make_colors_array(
    target_topologies: set[frozenset[frozenset[int]]],
    bicompatible_topologies: set[frozenset[frozenset[int]]] | None = None
)-> npt.NDArray:
    colors = []
    if bicompatible_topologies:
        for i in range(len(target_topologies)):
            colors.append("red")
        for i, topology in enumerate(target_topologies):
            if topology in bicompatible_topologies:
                colors[i] = 'g'
    else:
        for i in range(len(target_topologies)):
            colors.append("green")
    colors[0] = '#DAA520'
    colors[-1] = 'm'
    return colors


def reverse_positions_by_y_coordinate(
    positions: dict[tuple[int, int]]
)-> None:
    y_max = 0
    for pos in positions.values():
        if pos[1] > y_max:
            y_max = pos[1]
    for key in range(len(positions)):
        positions[key] = (positions[key][0], y_max - positions[key][1])


def is_distributive(
    topologies: set[frozenset[frozenset[int]]],
)-> None:
    flag = True
    for top1 in topologies:
        if not flag: 
            break
        for top2 in topologies:
            if not flag:
                break
            for top3 in topologies:
                inf_sup = topological_infimum(topological_supremum(top1, top2), top3)
                sup_inf_inf = topological_supremum(topological_infimum(top1, top3), topological_infimum(top2, top3))
                if inf_sup == sup_inf_inf:
                    continue
                else:
                    flag = False
    return flag

def is_modular(
    topologies: set[frozenset[frozenset[int]]]
)-> None:
    flag = True
    for top1 in topologies:
        if not flag: break
        for top2 in topologies:
            if not flag: break
            for top3 in topologies:
                if top3 <= top2:
                    if (topological_supremum(top3, topological_infimum(top1, top2)) != 
                        topological_infimum(topological_supremum(top3, top1), top2)):
                        flag = False
    return flag