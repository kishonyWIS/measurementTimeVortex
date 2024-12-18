import numpy as np
from lll_reduce import lll_reduce_2d
import matplotlib.pyplot as plt
from tqdm import tqdm

def is_valid_torus(L1: tuple[int,int,int], L2: tuple[int,int,int]):
    a1, b1, t1 = L1
    a2, b2, t2 = L2
    n1 = int(t1/6)
    n2 = int(t2/6)
    A = get_area(L1, L2, signed=True)
    if A == 0:
        return False
    val1 = 3 * (n1*b2 - n2*b1) / A
    val2 = 3 * (n1*(-b2-a2) + n2*(b1+a1)) / A
    val3 = 3 * (n1*a2 - n2*a1) / A
    # check if all three values are between -1 and 5
    return all([-1 < val < 5 for val in [val1, val2, val3]])

def get_area(L1: tuple[int,int,int], L2: tuple[int,int,int], signed=False):
    a1, b1, t1 = L1
    a2, b2, t2 = L2
    area = 2 * (a1*b2 - a2*b1)
    return area if signed else abs(area)

def get_vector_length(v: tuple[int,int,int]):
    w = (1/12*np.array(v) @
         np.array([[-8,4,4],[-4,8,-4],[1,1,1]]))
    return int(1/2*(abs(w[0])+abs(w[1])+abs(w[2])+abs(w[0]+w[1]+w[2])))

def lll_reduce(L1: tuple[int,int,int], L2: tuple[int,int,int]):
    B = np.array([L1, L2])
    reduced_B = lll_reduce_2d(B)
    return tuple(reduced_B)

def get_distance_torus(L1: tuple[int,int,int], L2: tuple[int,int,int]):
    d = np.inf
    max_m = 2
    for m1 in range(-max_m,max_m+1):
        for m2 in range(-max_m,max_m+1):
            if m1 == 0 and m2 == 0:
                continue
            v = m1*np.array(L1) + m2*np.array(L2)
            d = min(d, get_vector_length(v))
    return d, L1, L2

def generate_all_lattices(max_vars):
    unique_lattices = set()
    for c1 in tqdm(range(0,max_vars+1)):
        for c2 in range(-max_vars,max_vars+1):
            for d1 in range(0,max_vars+1):
                for d2 in range(-max_vars,max_vars+1):
                    for n1 in range(-max_vars,max_vars+1):
                        for n2 in range(-max_vars,max_vars+1):
                            L1 = (c1+3*d1, c1, 6*n1)
                            L2 = (c2+3*d2, c2, 6*n2)
                            if is_valid_torus(L1, L2):
                                L1, L2 = lll_reduce(L1, L2)
                                L1, L2 = tuple(map(int, L1)), tuple(map(int, L2))
                                unique_lattices.add((L1, L2))
    return list(unique_lattices)


if __name__ == '__main__':
    # find the torus with the best area for each distance
    best_area_for_dist = dict()
    best_torus_for_dist = dict()
    best_area_for_dist_no_vortex = dict()
    best_torus_for_dist_no_vortex = dict()
    max_vars = 20
    for L1, L2 in tqdm(generate_all_lattices(max_vars)):
        # filter even number of vortices in each direction
        if L1[-1] % 12 != 0 or L2[-1] % 12 != 0:
            continue
        dist, L1, L2 = get_distance_torus(L1, L2)
        area = get_area(L1, L2)
        if dist not in best_area_for_dist or area < best_area_for_dist[dist]:
            best_area_for_dist[dist] = area
            best_torus_for_dist[dist] = (L1, L2)
        if L1[-1] == 0 and L2[-1] == 0:
            if dist not in best_area_for_dist_no_vortex or area < best_area_for_dist_no_vortex[dist]:
                best_area_for_dist_no_vortex[dist] = area
                best_torus_for_dist_no_vortex[dist] = (L1, L2)
    # print the results
    for dist in sorted(best_area_for_dist.keys()):
        print(f'distance: {dist}, area: {best_area_for_dist[dist]}, A/d^2: {best_area_for_dist[dist]/dist**2}, torus: {best_torus_for_dist[dist]}')
    for dist in sorted(best_area_for_dist_no_vortex.keys()):
        print(f'distance: {dist}, area: {best_area_for_dist_no_vortex[dist]}, A/d^2: {best_area_for_dist_no_vortex[dist]/dist**2}, torus: {best_torus_for_dist_no_vortex[dist]}')
    # plot the area/distance squared as a function of distance
    x = sorted(best_area_for_dist.keys())
    y = [best_area_for_dist[dist]/dist**2 for dist in x]
    plt.plot(x, y)
    x = sorted(best_area_for_dist_no_vortex.keys())
    y = [best_area_for_dist_no_vortex[dist]/dist**2 for dist in x]
    plt.plot(x, y)
    plt.xlabel('$d$')
    plt.ylabel('$A/d^2$')
    plt.show()