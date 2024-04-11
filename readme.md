# Use numpy to calculate one-to-many iou

## Known issues / Unfinished items

* partial cell implementation for query_from_map
* speedup for integer locations

## Requirements

```bash
conda create -n iou python=3.11
conda activate iou
pip install numpy
pip install ipython
```

## Format

* one bounding box: np.array with shape `(4, )`
* n bounding boxes: np.array with shape `(n, 4)`
* each bounding box: `xmin, ymin, xmax, ymax`
* type: floating point
* tolerance: we expect the distance of two different floating number should be larger than `tol``

## Naive methods and test data generation

```ipython
from np_naive import gen_q_db, onemanyiou

q, db = gen_q_db(128)

%timeit onemanyiou(q, db)

ious = onemanyiou(q, db)
```

## Create map for query

```ipython
import numpy as np

from np_naive import gen_q_db, onemanyiou
from np_map_method import create_map, query_from_map

q, db = gen_q_db(128)

%timeit onemanyiou(q, db)
%timeit create_map(db)


ious1 = onemanyiou(q, db)

xlist, ylist, maplist, maparea, boxarea, floor_invx, floor_invy = create_map(db)

%timeit query_from_map(q, xlist, ylist, maplist, maparea, boxarea, floor_invx, floor_invy, accurate=False)


ious2 = query_from_map(q, xlist, ylist, maplist, maparea, boxarea, floor_invx, floor_invy, accurate=False)
np.max(np.abs(ious1 - ious2))

ious3 = query_from_map(q, xlist, ylist, maplist, maparea, boxarea, floor_invx, floor_invy, accurate=True)
%timeit query_from_map(q, xlist, ylist, maplist, maparea, boxarea, floor_invx, floor_invy, accurate=True)
np.max(np.abs(ious1 - ious3))
```