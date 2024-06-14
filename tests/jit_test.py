import numpy as np
from numba import jit
import time

@jit(nopython=True, cache=True, parallel=True)
def stack_array(x, shape):
    b = np.zeros(shape)
    for i in range(600):
        b[i] = x[i]
    return b

if __name__== "__main__":
    bs = 600
    sh = (3,256,256,4)
    a = []
    for i in range(bs):
        a.append(np.random.rand(*sh))
    shape = (bs, *sh)
    start = time.time()
    b = stack_array(a, shape)
    print("time:", time.time()-start)
