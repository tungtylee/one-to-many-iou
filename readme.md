# Use numpy to calculate one-to-many iou


## Requirements

```bash
conda create -n iou python=3.11
conda activate iou
pip install numpy
pip install ipython
```

## format

* one bounding box: np.array with shape `(4, )`
* n bounding boxes: np.array with shape `(n, 4)`
* each bounding box: `xmin, ymin, xmax, ymax`
* type: floating point
* tolerance: we expect the distance of two different floating number should be larger than `tol``

## Naive methods and test data generation

```ipython
from np_naive import gen_q_db, onemanyiou

q, db = gen_q_db(1024)

%timeit onemanyiou(q, db)

ious = onemanyiou(q, db)
```
