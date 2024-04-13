import timeit

import numpy as np

from np_map_method import create_map, query_from_map
from np_naive import gen_q_db, onemanyiou


def naive_method__onemanyiou():
    onemanyiou(q, db)


def map_method__create_map():
    xlist, ylist, maplist, maparea, boxarea, floor_invx, floor_invy = create_map(db)


def map_method__query_from_map0():
    query_from_map(
        q,
        xlist,
        ylist,
        maplist,
        maparea,
        boxarea,
        floor_invx,
        floor_invy,
        accurate=False,
    )


def map_method__query_from_map1():
    query_from_map(
        q,
        xlist,
        ylist,
        maplist,
        maparea,
        boxarea,
        floor_invx,
        floor_invy,
        accurate=True,
    )


### Data Preparation

q, db = gen_q_db(128)

### Accuracy
print("Accuracy Report:")
ious1 = onemanyiou(q, db)
xlist, ylist, maplist, maparea, boxarea, floor_invx, floor_invy = create_map(db)
ious2 = query_from_map(
    q, xlist, ylist, maplist, maparea, boxarea, floor_invx, floor_invy, accurate=False
)
ious3 = query_from_map(
    q, xlist, ylist, maplist, maparea, boxarea, floor_invx, floor_invy, accurate=True
)
print("Max abs diff (fast)", np.max(np.abs(ious1 - ious2)))
print("Max abs diff (accurate)", np.max(np.abs(ious1 - ious3)))


### Execution Time (for 100 iterations)
print("Execution Time for 100 iterations:")

time_result = timeit.timeit("naive_method__onemanyiou()", globals=globals(), number=100)
print("naive_method__onemanyiou", time_result)

time_result = timeit.timeit("map_method__create_map()", globals=globals(), number=100)
print("map_method__create_map", time_result)

time_result = timeit.timeit(
    "map_method__query_from_map0()", globals=globals(), number=100
)
print("map_method__query_from_map_fast", time_result)

time_result = timeit.timeit(
    "map_method__query_from_map1()", globals=globals(), number=100
)
print("map_method__query_from_map_accurate", time_result)
