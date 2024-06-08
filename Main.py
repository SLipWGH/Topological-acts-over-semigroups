

import networkx as nx

from time import time

import matplotlib.pyplot as plt

from Algebraic import Operation, boundedsum_of_two_2, bounded_sum_of_two_muls_by_mod_4

from Topology import get_all_topologies, find_topologies_operation_compatible
from Lattices import get_graph_by_topologies, make_colors_array, reverse_positions_by_y_coordinate, get_topologies_list_ordered_by_power, is_distributive, is_modular




def disjunction_compatible_task(
    X: set,
    topologies: set[frozenset[frozenset[int]]]
)-> None:
    start = time()
    disjunction_compatible_topologies = find_topologies_operation_compatible(X, topologies, Operation(max, 2))
    print(f"time to find target topologies = {time() - start}")

    bicompatible_topologies = find_topologies_operation_compatible(X, disjunction_compatible_topologies, Operation(min, 2))

    print(len(bicompatible_topologies))
    print(f"time to find boolean = {time() - start}")

    is_lattice_modular = is_modular(disjunction_compatible_topologies)

    print(f"is lattice modular: {is_lattice_modular}")

    if is_lattice_modular:
        print(f"is lattice distributive{is_distributive(disjunction_compatible_topologies)}")

    disjunction_compatible_topologies = get_topologies_list_ordered_by_power(disjunction_compatible_topologies)
    

    G = get_graph_by_topologies(disjunction_compatible_topologies)
    colors = make_colors_array(disjunction_compatible_topologies, bicompatible_topologies)
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    reverse_positions_by_y_coordinate(pos)

    nx.draw(G, pos=pos, node_color=colors, with_labels=True, alpha = 0.9,)
    plt.show()



def custom_operation_task(
    X: set,
    topologies: set[frozenset[frozenset[int]]]  
)-> None:

    operation = Operation(bounded_sum_of_two_muls_by_mod_4, 4)

    start = time()
    target_topologies = get_topologies_list_ordered_by_power(
        find_topologies_operation_compatible(X, topologies, operation))

    print(f"time to find target topologies = {time() - start}")

    is_lattice_modular = is_modular(target_topologies)

    print(f"is lattice modular: {is_lattice_modular}")

    if is_lattice_modular:
        print(f"is lattice distributive {is_distributive(target_topologies)}")

    G = get_graph_by_topologies(target_topologies)
    colors = make_colors_array(target_topologies)
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    reverse_positions_by_y_coordinate(pos)
    nx.draw(G, pos=pos, node_color=colors, with_labels=True, alpha = 0.9,)
    plt.show()


def main(
    disjunction_task = True
)-> None:
    n = 6
    X_op = {i for i in range(n)}
    start = time()
    topologies = get_all_topologies(n)
    print(f"time t = = {time() - start} to find all topologies on X = {X_op} ")

    if disjunction_task:
        disjunction_compatible_task(X_op, topologies)
    else:
        custom_operation_task(X_op, topologies)


if __name__ == "__main__":
    main() 
