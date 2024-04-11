import numpy as np


def gen_q_db(n):
    query_data = np.random.randint(0, 6400, 2).tolist() + np.random.randint(0, 4800, 2).tolist()
    query_data = query_data
    query_data[2] += query_data[0]  # Ensure xmax > xmin
    query_data[3] += query_data[1]  # Ensure ymax > ymin

    q = np.array(query_data)
    q = q / 6.0

    # Generate a database (db) of 32 bounding boxes with the specified constraints
    db = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        xmin, xmax = np.random.randint(0, 6400, 2)
        ymin, ymax = np.random.randint(0, 4800, 2)
        xmin, xmax = sorted([xmin, xmax])
        ymin, ymax = sorted([ymin, ymax])
        db[i] = [xmin, ymin, xmax, ymax]
    db = db / 6.0
    return q, db
  
def onemanyiou(query, db):
    # a query is a bounding box xmin, ymin, xmax, ymax with shape (4,)
    # a db is a n bounding boxes with shape (n, 4)
    # return ious with shape (n,)

    xi1 = np.maximum(query[0], db[:, 0])
    yi1 = np.maximum(query[1], db[:, 1])
    xi2 = np.minimum(query[2], db[:, 2])
    yi2 = np.minimum(query[3], db[:, 3])
    
    inter_area = np.maximum(xi2 - xi1, 0) * np.maximum(yi2 - yi1, 0)
    
    # Compute each area
    query_area = (query[2] - query[0]) * (query[3] - query[1])
    db_areas = (db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1])
    
    # Compute union
    union_area = query_area + db_areas - inter_area
    
    # Compute IoU
    ious = inter_area / union_area
    
    return ious