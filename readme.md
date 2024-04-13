# Use numpy to calculate one-to-many iou

## Known issues / Unfinished items

- [x] partial cell implementation for query_from_map
- [ ] not handle query box width or height less than original resolution
- [ ] floating point is too slow, speedup for integer locations

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

## Report

```bash
python test.py
```

```text
Accuracy Report:
Max abs diff (fast) 0.003242057242958514
Max abs diff (accurate) 6.877390308434109e-05
Execution Time for 100 iterations:
naive_method__onemanyiou 0.0017145780002465472
map_method__create_map 22.701142375000018
map_method__query_from_map_fast 0.07468162900022435
map_method__query_from_map_accurate 0.08704350199968758
```


## Naive methods and test data generation

```ipython
from np_naive import gen_q_db, onemanyiou

q, db = gen_q_db(128)

%timeit onemanyiou(q, db)

ious1 = onemanyiou(q, db)
```

## Create map for query and two query methods

```ipython
import numpy as np

from np_naive import gen_q_db, onemanyiou
from np_map_method import create_map, query_from_map

q, db = gen_q_db(128)

ious2 = query_from_map(q, xlist, ylist, maplist, maparea, boxarea, floor_invx, floor_invy, accurate=False)
ious3 = query_from_map(q, xlist, ylist, maplist, maparea, boxarea, floor_invx, floor_invy, accurate=True)
```