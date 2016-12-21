import numpy as np
from scripts.utils import parse_dataset

np.set_printoptions(suppress=True)

dataset = None
X, y, w, ids = parse_dataset(cls=dataset)
rows, cols = X.shape

def ams(y, w):
    s = w[y==1.].sum()
    b = 0
    b_reg = 10

    return np.sqrt(2*((s+b+b_reg)*np.log(s/(b+b_reg)+1)-s))

print(ams(y, w))