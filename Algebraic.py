
from itertools import product
from typing import Callable, Iterable



class Operation:
    def __init__(
        self,
        operation: Callable,
        arity: int
    ) -> None:
        self._operation = operation
        self._arity = arity


    def __getitem__(
        self, 
        operands: Iterable
    )-> frozenset[int]:
        if len(operands) != self._arity:
            raise ValueError(f'{self._arity} operands must be passed')
        if all([isinstance(element, int) for element in operands]):
            return self._operation(*map(int, *product(*[[element] for element in operands]))) 
        elif not all([isinstance(item, frozenset) for item in operands]):
            raise ValueError("All operands must be frozensets of int or int")
        return frozenset([self._operation(*args) for args in product(*operands)])
        
        
    @property
    def arity(
        self
    )-> int:
        return self._arity
    

def bounded_sum_of_two_muls_by_mod_4(
    x: int,
    y: int,
    z: int,
    t: int,
)-> int:
    n  = 5
    res = x * y % (n + 1) + z * t % (n + 1)
    return res if res < n else n


def boundedsum_of_two_2(
    x: int,
    y: int,
    # Xmax: int
)-> int:
    n  = 5
    return x + y if x + y < n else n


