import matplotlib.pyplot as plt

from color_code import *
from lattice import *


for num_vortexes in [(0,0), (1,0), (0,1)]:
    lat = HexagonalLattice((6,0),(0,6))
    code = FloquetCode(lat, num_vortexes=num_vortexes, detectors=('Z',))

    # with sns.axes_style("white"):
    plt.rcParams["font.family"] = "Times New Roman"
    # change the font size
    for with_colorbar in [True, False]:
        code.draw_pauli(None, color_bonds_by_delay=num_vortexes!=(0,0), show=False, fontsize_measurements=0, linewidth=10, qubitsize=20, colorbar_flag=with_colorbar, rotate_colorbar=120)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(
            f'figures/color_code_num_vortexes_{num_vortexes}_colorbar_{with_colorbar}.pdf', bbox_inches='tight', pad_inches=0)
plt.show()