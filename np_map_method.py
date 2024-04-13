from collections import defaultdict

import numpy as np


def create_map(db, tol=0, extremeAdd=[0, 1200]):
    # db: multiple bounding boxes in a np.array with shape (n, 4)
    # return:
    #  xlist: np.array with shape (nx, )
    #  ylist: np.array with shape (ny, )
    #  maplist is Dict[Tuple[int, int], List[int]]
    #  maparea is Dict[Tuple[int, int], float]
    #  boxarea is a np.array with shape (n, )
    #  floor_invx is Dict[int, int]
    #  floor_invy is Dict[int, int]

    # unique and sorted
    # TODO
    # dbvalplus = np.concatenate((db[:, 0::2], np.array(extremeAdd).reshape(-1, 2)))
    xlist = np.unique(db[:, 0::2])
    xlist.sort()
    # dbvalplus = np.concatenate((db[:, 1::2], np.array(extremeAdd).reshape(-1, 2)))
    ylist = np.unique(db[:, 1::2])
    ylist.sort()
    boxarea = (db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1])
    nx = len(xlist)
    ny = len(ylist)
    maplist = defaultdict(list)
    maparea = {}
    floor_invx = {}
    floor_invy = {}

    for idx, val in enumerate(xlist):
        tag = int(np.floor(val))
        if tag not in floor_invx:
            floor_invx[tag] = idx
    for idx, val in enumerate(ylist):
        tag = int(np.floor(val))
        if tag not in floor_invy:
            floor_invy[tag] = idx

    for idx, val in enumerate(db):
        x0, y0, x1, y1 = val
        startx = floor_invx[int(np.floor(x0))]
        starty = floor_invy[int(np.floor(y0))]

        fx0 = startx
        fx1 = nx
        fy0 = starty
        fy1 = ny
        for ix in range(startx, nx):
            if x0 > xlist[ix]:
                fx0 = ix + 1
                continue
            elif x0 <= xlist[ix] and x1 > xlist[ix]:
                fx1 = ix
            else:
                fx1 = ix
                break
        for iy in range(starty, ny):
            if y0 > ylist[iy]:
                fy0 = iy + 1
                continue
            elif y0 <= ylist[iy] and y1 > ylist[iy]:
                fy1 = iy
            else:
                fy1 = iy
                break

        for ix in range(fx0, fx1):
            for iy in range(fy0, fy1):
                maplist[(ix, iy)].append(idx)
                if (ix, iy) not in maparea:
                    maparea[(ix, iy)] = (xlist[ix + 1] - xlist[ix]) * (
                        ylist[iy + 1] - ylist[iy]
                    )

        # print(fx0, fx1, fy0, fy1)
        # print(xlist[fx0], xlist[fx1], ylist[fy0], ylist[fy1])
        # print(db[idx])

    return xlist, ylist, maplist, maparea, boxarea, floor_invx, floor_invy


def query_from_map(
    q, xlist, ylist, maplist, maparea, boxarea, floor_invx, floor_invy, accurate=True
):
    # q: multiple bounding boxes in a np.array with shape (n, 4)
    # xlist: np.array with shape (nx, )
    # ylist: np.array with shape (ny, )
    # maplist is Dict[Tuple[int, int], List[int]]
    # maparea is Dict[Tuple[int, int], float]
    # boxarea is a np.array with shape (n, )
    # floor_invx is Dict[int, int]
    # floor_invy is Dict[int, int]
    # return
    #  ious
    nb = boxarea.shape[0]
    interarea = np.zeros(nb)
    nx = len(xlist)
    ny = len(ylist)

    x0, y0, x1, y1 = q
    qarea = (x1 - x0) * (y1 - y0)
    fx0 = 0
    fx1 = nx
    fy0 = 0
    fy1 = ny
    for ix in range(nx):
        if x0 > xlist[ix]:
            fx0 = ix + 1
            continue
        elif x0 <= xlist[ix] and x1 > xlist[ix]:
            fx1 = ix
        else:
            fx1 = ix
            break
    for iy in range(ny):
        if y0 > ylist[iy]:
            fy0 = iy + 1
            continue
        elif y0 <= ylist[iy] and y1 > ylist[iy]:
            fy1 = iy
        else:
            fy1 = iy
            break
    # handle cells
    for ix in range(fx0, fx1):
        for iy in range(fy0, fy1):
            for otherb in maplist[(ix, iy)]:
                interarea[otherb] += maparea[(ix, iy)]

    if accurate is False:
        ious = interarea / (boxarea - interarea + qarea)
        return ious

    # handle small cells
    # TODO

    # handle partial cells
    # 012
    # 3x5
    # 678
    if x0 < xlist[fx0] and y0 < ylist[fy0]:
        # case 0
        smallarea = (xlist[fx0] - x0) * (ylist[fy0] - y0)
        assert smallarea >= 0
        for otherb in maplist[(fx0 - 1, fy0 - 1)]:
            interarea[otherb] += smallarea
    if y0 < ylist[fy0]:
        # case 1
        for ix in range(fx0, fx1):
            smallarea = (xlist[ix + 1] - xlist[ix]) * (ylist[fy0] - y0)
            assert smallarea >= 0
            for otherb in maplist[(ix, fy0 - 1)]:
                interarea[otherb] += smallarea
    if fx1 != nx and x1 > xlist[fx1] and y0 < ylist[fy0]:
        # case 2
        smallarea = (x1 - xlist[fx1]) * (ylist[fy0] - y0)
        assert smallarea >= 0
        for otherb in maplist[(fx1, fy0 - 1)]:
            interarea[otherb] += smallarea

    if x0 < xlist[fx0]:
        # case 3
        for iy in range(fy0, fy1):
            smallarea = (xlist[fx0] - x0) * (ylist[iy + 1] - ylist[iy])
            assert smallarea >= 0
            for otherb in maplist[(fx0 - 1, iy)]:
                interarea[otherb] += smallarea
    if fx1 != nx and x1 > xlist[fx1]:
        # case 5
        for iy in range(fy0, fy1):
            smallarea = (x1 - xlist[fx1]) * (ylist[iy + 1] - ylist[iy])
            assert smallarea >= 0
            for otherb in maplist[(fx1, iy)]:
                interarea[otherb] += smallarea

    if x0 < xlist[fx0] and fy1 != ny and y1 > ylist[fy1]:
        # case 6
        smallarea = (xlist[fx0] - x0) * (y1 - ylist[fy1])
        assert smallarea >= 0
        for otherb in maplist[(fx0 - 1, fy1)]:
            interarea[otherb] += smallarea
    if fy1 != ny and y1 > ylist[fy1]:
        # case 7
        for ix in range(fx0, fx1):
            smallarea = (xlist[ix + 1] - xlist[ix]) * (y1 - ylist[fy1])
            assert smallarea >= 0
            for otherb in maplist[(ix, fy1)]:
                interarea[otherb] += smallarea
    if fx1 != nx and x1 > xlist[fx1] and fy1 != ny and y1 > ylist[fy1]:
        # case 8
        smallarea = (x1 - xlist[fx1]) * (y1 - ylist[fy1])
        assert smallarea >= 0
        for otherb in maplist[(fx1, fy1)]:
            interarea[otherb] += smallarea
    ious = interarea / (boxarea - interarea + qarea)
    return ious
