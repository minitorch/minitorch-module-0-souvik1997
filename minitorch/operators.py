"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Any, Optional

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

def mul(a: Any, b: Any) -> Any:
    return a * b

def id(a: Any):
    return a

def add(a: Any, b: Any) -> Any:
    return a + b

def neg(a: Any) -> Any:
    return -a

def lt(a: Any, b: Any) -> bool:
    return a < b

def eq(a: Any, b: Any) -> bool:
    return a == b

def max(a: Any, b: Any) -> Any:
    return b if lt(a, b) else a

def abs(a: Any) -> Any:
    if lt(a, 0):
        return neg(a)
    else:
        return a

def is_close(a: Any, b: Any) -> bool:
    return lt(abs(add(a, neg(b))), 1e-2)

def sigmoid(a: Any) -> Any:
    if lt(a, 0):
        return mul(exp(a), inv(1.0 + exp(a)))
    else:
        return mul(1.0, inv(1.0 + exp(neg(a))))
    
def relu(a: Any) -> Any:
    if lt(a, 0):
        return 0
    else:
        return a

def log(a: Any) -> Any:
    return math.log(a)

def exp(a: Any) -> Any:
    return math.exp(a)

def log_back(a: Any, b: Any) -> Any:
    return mul(b, inv(a))

def inv(a: Any) -> Any:
    return 1.0 / a

def inv_back(a: Any, b: Any) -> Any:
    return mul(b, neg(mul(inv(a), inv(a))))

def relu_back(a: Any, b: Any) -> Any:
    if lt(a, 0):
        return 0
    else:
        return b
    

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

def map[T, U](f: Callable[[T], U], a: Iterable[T]) -> Iterable[U]:
    for x in a:
        yield f(x)

def zipWith[T, U, V](f: Callable[[T, U], V], a: Iterable[T], b: Iterable[U]) -> Iterable[V]:
    a_i = iter(a)
    b_i = iter(b)
    while True:
        try:
            a_n = next(a_i)
            b_n = next(b_i)
            yield f(a_n, b_n)
        except StopIteration:
            break

def reduce[T](f: Callable[[T, T], T], a: Iterable[T], initial: Optional[T] = None) -> T:
    a_i = iter(a)
    if initial is None:
        try:
            result = next(a_i)
        except StopIteration:
            raise TypeError("reduce() of empty iterable with no initial value")
    else:
        result = initial
    
    while True:
        try:
            a_n = next(a_i)
            result = f(result, a_n)
        except StopIteration:
            break

    return result

def negList[T](a: Iterable[T]) -> Iterable[T]:
    return map(neg, a)

def addLists[T](a: Iterable[T], b: Iterable[T]) -> Iterable[T]:
    return zipWith(add, a, b)

def sum[T](a: Iterable[T]) -> T:
    return reduce(add, a, 0)

def prod[T](a: Iterable[T]) -> T:
    return reduce(mul, a, 1)
