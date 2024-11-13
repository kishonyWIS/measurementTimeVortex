import numpy as np
import matplotlib.pyplot as plt

def is_allowed_vortex_numbers(Lx, Ly, vx, vy):
    condition1 = -(vy+1/2) - (3*Ly - 2)/(2*Lx - 1) * abs(vx)+1/2 + 2*Ly
    condition2 = vy + Ly
    print(f'condition1={condition1}, condition2={condition2}')
    return condition1>=0 and condition2>0

Lx = 2
Ly = 3
vx_list = np.arange(-5, 6)
vy_list = np.arange(-5, 6)
allowed = np.zeros((len(vx_list), len(vy_list)), dtype=bool)
for i, vx in enumerate(vx_list):
    for j, vy in enumerate(vy_list):
        allowed[i, j] = is_allowed_vortex_numbers(Lx, Ly, vx, vy)

plt.pcolor(allowed.T)
plt.xticks(np.arange(len(vx_list))+0.5, vx_list)
plt.yticks(np.arange(len(vy_list))+0.5, vy_list)
plt.xlabel('vx')
plt.ylabel('vy')
plt.show()