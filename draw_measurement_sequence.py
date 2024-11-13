import matplotlib.pyplot as plt

from color_code import *
from lattice import *

d = (2,3)

for num_vortexes in [(0,0), (1,0), (0,1)]:
    lat = HexagonalLatticeGidney(d)
    code = FloquetCode(lat, num_vortexes=num_vortexes, detectors=('Z',))

    # with sns.axes_style("white"):
    plt.rcParams["font.family"] = "Times New Roman"
    # change the font size
    code.draw_pauli(None, False, show=False, fontsize_measurements=17)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(
        f'figures/color_code_num_vortexes_{num_vortexes}.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()