import stim
import numpy as np
converter = stim.Circuit('''
X 0
M 0
DETECTOR rec[-1]
''').compile_m2d_converter()
new = converter.convert(measurements=np.array([[0], [1]], dtype=np.bool8), append_observables=False)
print(new)