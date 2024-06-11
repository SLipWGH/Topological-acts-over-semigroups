from itertools import product
from multiprocessing import Process, Queue, Manager, cpu_count
from typing import Iterable

import queue

import numpy as np
import numpy.typing as npt
import numba as nb

from Algebraic import Operation


def get_powerset(
    n: int
)-> npt.NDArray[np.int8]:
    """returns powerset in decimal representation"""
    return np.arange(2**n, dtype=np.int8)


def get_atomic(
    n: int,
)-> npt.NDArray[np.int8]:
    """returns list of atomic elements in decimal representation"""
    res = np.roll(np.full((n,), 2, dtype=np.int8).cumprod(dtype=np.int8), 1)
    res[0] = 1
    return res


def get_atomic_decomposition(
    x: int,
    n: int
)-> npt.NDArray[np.int8]:
    """returns decomposition into atomic of x as list of bits indexes included in x"""
    at = get_atomic(n)
    return np.where(at & x == at)


def get_images_saveing_inclusion_property(
    x: int,
    n: int
)-> npt.NDArray[np.int8]:
    """returns list with potential images of x saveing inclusion property."""
    ps = get_powerset(n)
    return ps[ps & x == x]


def get_potential_images_matrix(
    n: int
)->npt.NDArray[np.int8]:
    """returns matrix with potential images for every atomic element of powerset"""
    return np.vstack([get_images_saveing_inclusion_property(i, n) for i in get_atomic(n)], dtype=np.int8)


@nb.njit
def is_not_available_image(
    atomic: np.int8,
    image: np.int8,
    previous_atomic: np.int8,
    previous_image: np.int8
)-> bool:
    """conditions aimed at preserving the idempotency of the operator. Returns True if image doesnt ruins idempotency of operator."""
    condition_1 = atomic & previous_image == atomic and image | previous_image != previous_image
    condition_2 = previous_atomic & image == previous_atomic and image | previous_image != image
    return condition_1 or condition_2


@nb.njit
def get_restricted_matrix(
    undefined_atomic_elements_inverted: npt.NDArray[np.int8],
    previous_defined_atomic: np.int8,
    previous_atomic_image: np.int8,
    PIms: npt.NDArray[np.int8]
)-> npt.NDArray[np.int8]:
    """returns restricted matrix of potential images for all other atomic, saveing idempotency property of operator"""
    if undefined_atomic_elements_inverted.size == 0:
        return None
    
    for atomic, i in zip(undefined_atomic_elements_inverted, range(len(PIms))):
        for j in range(len(PIms[i])):
            if is_not_available_image(atomic, PIms[i][j], previous_defined_atomic, previous_atomic_image):
                PIms[i][j] = 0
    return PIms


@nb.njit(parallel = True)
def find_closure_preoperators_recursively(
    collector: npt.NDArray[np.int8],
    PIms: npt.NDArray[np.int8] | None,
    atomic_undefined: npt.NDArray[np.int8],
    atomic_images: npt.NDArray[np.int8],
)-> None:
    currently_defined = atomic_undefined[0]

    for image in PIms[0]:
        if image == 0: continue
        map = atomic_images.copy()
        map[np.where(map == 0)[0][0]] = image
        restricted_PIms = get_restricted_matrix(
            atomic_undefined[1:], 
            currently_defined,
            image, 
            PIms[1:].copy()
        )
        if restricted_PIms is not None:
            collector = find_closure_preoperators_recursively(
                    collector, 
                    restricted_PIms, 
                    atomic_undefined[1:],  
                    map
                )
        else:
            collector=np.append(collector, map)
    return collector


def get_closure_preoperators(
    n: int
)-> list:
    atomic_set = get_atomic(n)
    PIms = get_potential_images_matrix(n)
    closure_preoperators_collection = np.array([], dtype=np.int8)
    closure_preoperators_collection = find_closure_preoperators_recursively(
        collector=closure_preoperators_collection, 
        PIms=PIms, 
        atomic_undefined=atomic_set,
        atomic_images=np.zeros_like(atomic_set)
    )
    closure_preoperators_collection.resize((closure_preoperators_collection.size // atomic_set.size, atomic_set.size))
    return closure_preoperators_collection


@nb.njit
def continue_closure_preoperator_to_operator(
    closure: npt.NDArray[np.int8]
)-> npt.NDArray[np.int8]:
    prev = np.empty_like(closure)
    while not np.array_equal(prev, closure):
        prev = closure.copy()
        for a in prev:
            for b in prev:
                c = np.int8(a & b)
                d = np.int8(a | b)
                if np.where(closure == c)[0].size == 0: 
                    closure = np.append(closure, c)
                if np.where(closure == d)[0].size == 0:
                    closure = np.append(closure, d)
    return closure


def int_to_frozenset(
    n: int,
    power: int
)-> set:
    atomic = get_atomic_decomposition(n, power)
    return frozenset(*atomic)


def contruct_topology_by_closure_operator(
    closed: set[int],
    n: int
)-> set[frozenset]:
    topology = set()
    for close in closed:
        topology.add(frozenset(int_to_frozenset(close, n)))
    topology.add(frozenset())
    return topology


def get_all_topologies(
    n: int,
)-> set[frozenset[frozenset[int]]]: 
    topologies = set()
    for cl in get_closure_preoperators(n):
        closure_operator = continue_closure_preoperator_to_operator(cl)
        topologies.add(frozenset(contruct_topology_by_closure_operator(closure_operator, n)))
    return topologies


def topological_infimum(
    top1: set[frozenset[int]],
    top2: set[frozenset[int]]
)-> set[frozenset[int]]:
    return top1 & top2


def topological_supremum(
    top1: set[frozenset[int]],
    top2: set[frozenset[int]]
)-> set[frozenset[int]]:
    prebase = get_prebase_by_two_topologies(top1, top2)
    base = get_base_by_prebase(prebase)
    return get_topology_by_base(base)


def get_prebase_by_two_topologies(
    top1: set[frozenset[int]],
    top2: set[frozenset[int]]
)-> set[frozenset[int]]:
    return set(top1 | top2)


def get_base_by_prebase(
    prebase: set[frozenset[int]]
)-> set[frozenset[int]]:
    buffer = set()
    while buffer != prebase:
        buffer.update(prebase)
        for p1 in buffer:
            for p2 in buffer:
                prebase.add(frozenset(p1 & p2))
    return prebase


def get_topology_by_base(
    base: set[frozenset[int]]
)-> set[frozenset[int]]:
    buffer = set()
    while buffer != base:
        buffer.update(base)
        for p1 in buffer:
            for p2 in buffer:
                base.add(frozenset(p1 | p2))
    return base


def work(
    tasks_to_perform: Queue, 
    complete_tasks: Queue,
    X: set,
    operation: Operation
)-> None:
    while True:
        try:
            task = tasks_to_perform.get_nowait()
        except queue.Empty:
            break
        else:
            res = is_topology_operation_compatible(X, task, operation)
            complete_tasks.put_nowait(res)
    return True 


def find_topologies_operation_compatible(
    X: set,
    topologies: set[frozenset[int]],
    operation: Operation
)-> set[frozenset]:
    # sort_key = lambda x: (len(x) ** (operation.arity + 1)) * np.prod([len(item) for item in x])
    topologies = sorted(topologies, key = len, reverse=True)
    result, topologies = [topologies[0]], topologies[1:]
    with Manager() as manager:
        tasks_to_perform = manager.Queue(len(topologies))  
        complete_tasks = manager.Queue(len(topologies))  
        processes: list[Process] = []  
        
        for topology in topologies:
            tasks_to_perform.put(topology)
        for _ in range(cpu_count()):
            p = Process(target=work, args=(tasks_to_perform, complete_tasks, X, operation))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        while not complete_tasks.empty():
            topology = complete_tasks.get()
            if topology is not None:
                result.append(topology)
    return set(result)


def found_nbhs_forall_nbh_of_product(
    points: list[int],
    topology: set[frozenset[int]],
    operation: Operation,
    points_product: int
)-> bool:
    points_product_neighbourhoods = (u for u in topology if points_product in u)
    for w in points_product_neighbourhoods:
        for u_i in product(*[[u for u in topology if point in u] for point in points]):
            if operation[u_i] <= w:
                break
        else: return False
    return True


def is_topology_operation_compatible(
    X: set,
    topology: set[frozenset[int]],
    operation: Operation
)-> set[frozenset[int]] | None:
    prod = [[*points] for points in product(X, repeat=operation.arity)]
    for points in prod:
        points_product = operation[points]
        if not found_nbhs_forall_nbh_of_product(points, topology, operation, points_product):
            return None
    return topology.copy()
