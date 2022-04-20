import numpy as np

from compute_trajectoid import intersects as intersects2
from great_circle_arc import intersects

def test_faster_intersection_algorithm():
    N = 2000
    K = 0
    for j in range(N):
        vectors = np.random.rand(4, 3)*2-1
        norms = np.linalg.norm(vectors, axis=1)
        for i in range(4):
            vectors[i, :] = vectors[i, :]/norms[i]
        intersection_found_by_intersects = intersects(vectors[0, :], vectors[1, :], vectors[2, :], vectors[3, :])
        if intersection_found_by_intersects:
            K += 1
        assert intersection_found_by_intersects == \
               intersects2(vectors[0, :], vectors[1, :], vectors[2, :], vectors[3, :])
    print(f' Probability of intersection: {K/N}')

### Speed test
# import time
# vectors = np.random.rand(4, 3)
# norms = np.linalg.norm(vectors, axis=1)
# for i in range(4):
#     vectors[i, :] = vectors[i, :]/norms[i]
#
# N = 100000
# t0 = time.time()
# for j in range(N):
#     X = intersects(vectors[0, :], vectors[1, :], vectors[2, :], vectors[3, :])
# print((time.time()-t0))
#
# t0 = time.time()
# for j in range(N):
#     X = intersects2(vectors[0, :], vectors[1, :], vectors[2, :], vectors[3, :])
# print((time.time()-t0))


##### Some old sketches:

# from numba import jit

# @jit(nopython=True)
# def numbacross(a,b):
#     return [a[1]*b[2] - b[1]*a[2], -a[0]*b[2] + b[0]*a[2], a[0]*b[1] - b[0]*a[1]]
#
# # @jit(nopython=True)
# # def crossx(a,b):
# #     return a[1]*b[2] - b[1]*a[2]
#
# @jit(nopython=True)
# def numbadotsign(a,b):
#     x = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
#     if x>0:
#         r = 1
#     elif x<0:
#         r = -1
#     else:
#         r = 0
#     return r
#
#
# # # @jit(nopython=True)
# # def intersects2(A, B, C, D):
# #     X = np.cross(np.cross(A, B), np.cross(C, D))
# #     # X = numbacross(numbacross(A, B), numbacross(C, D))
# #     # return (crossx(X, A)*crossx(X, B) < 0) and (crossx(X, C)*crossx(X, D) < 0)
# #     # optionB =
# #     optionA = (np.dot(np.cross(X, A), np.cross(X, B)) < 0 ) and (np.dot( np.cross(X, C), np.cross(X, D)) < 0 )
# #     optionB = (np.dot(np.cross(-X, A), np.cross(-X, B)) < 0) and (np.dot(np.cross(-X, C), np.cross(-X, D)) < 0)
# #     return optionA or optionB
#
# @jit(nopython=True)
# def intersects2(A, B, C, D):
#     ABX = numbacross(A, B)
#     CDX = numbacross(C, D)
#     T = numbacross(ABX, CDX)
#     s = 0
#     s += numbadotsign(numbacross(ABX, A), T)
#     s += numbadotsign(numbacross(B, ABX), T)
#     s += numbadotsign(numbacross(CDX, C), T)
#     s += numbadotsign(numbacross(D, CDX), T)
#     return (s == 4) or (s == -4)