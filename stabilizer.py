from __future__ import annotations

import galois
from qiskit import QiskitError
from qiskit.quantum_info import PauliList, Pauli, StabilizerState
import numpy as np


def pauli_weight(pauli: Pauli):
    return sum([str(p) != 'I' for p in pauli])


class StabilizerGroup(PauliList):
    """Class for representing a Pauli Stabilizer group.

    This class is a subclass of the PauliList class. It is used to represent
    a Pauli Stabilizer group, which is a list of Pauli objects. The class
    inherits the PauliList methods, and adds a few more methods that are
    specific to Pauli Stabilizer groups. A StabilizerGroup must have all Hermitian Pauli generators
    and must be abelian and must not include minus the identity.
    When the generators are not independent, the method reduce will
    reduce the number of generators to the minimum number of independent generators. The method measure_pauli
    will measure a Pauli operator, adding the result which is plus or minus that operator to the stabilizer group.
    It also checks whether any of the generators anticommute with the operator being measured, and if so, it removes
    one of these generators and multiplies all others by the one removed.
    """
    def __init__(self, *args, **kwargs):
        """Initialize a Pauli Stabilizer group.

        Args:
            *args: Variable length argument list for PauliList.
            **kwargs: Arbitrary keyword arguments for PauliList.
        """
        super().__init__(*args, **kwargs)
        # self._check_stabilizer_group()
        self._pauli_normalizer = None
        self._logical_Zs = None
        self._logical_Xs = None
        self.measurement_indexes = []

    def _check_stabilizer_group(self):
        """Check that the Pauli Stabilizer group is valid.

        Raises:
            QiskitError: If the Pauli Stabilizer group is not valid.
        """
        if not self.is_stabilizer_group():
            raise QiskitError('The Pauli Stabilizer group is not valid.')

    def is_stabilizer_group(self):
        """Check if the Pauli Stabilizer group is valid.

        Returns:
            bool: Whether the Pauli Stabilizer group is valid.
        """
        return self.is_abelian() and self.is_hermitian()

    def measure_pauli(self, pauli: Pauli, measurement_index: int):
        """Measure a Pauli operator and add the result to the stabilizer group. The method measure_pauli
        will measure a Pauli operator, adding the result which is plus or minus that operator to the stabilizer group.
        It also checks whether any of the generators anticommute with the operator being measured, and if so, it removes
        one of these generators and multiplies all others by the one removed. Finally, it reduces the number of generators
        to the minimum number of independent generators.

        Args:
            pauli (Pauli): The Pauli operator or a string to measure.

        Raises:
            QiskitError: If the Pauli operator is not a Pauli operator or if it is not a Pauli operator on the same
            number of qubits as the stabilizer group.
        """
        if not isinstance(pauli, Pauli):
            pauli = Pauli(pauli)
        if pauli.num_qubits != self.num_qubits:
            raise QiskitError('The Pauli operator to measure must be a Pauli operator on the same number of qubits as '
                              'the stabilizer group.')
        new = StabilizerGroup(self.copy())
        if new.contains_pauli(pauli, ignore_sign=True):
            return new
        new = new.append(pauli)
        new.measurement_indexes.append(measurement_index)
        # Check if any of the generators anticommute with the operator being measured
        generators_anticommuting_with_pauli = new.anticommutes_with_all(pauli)
        if len(generators_anticommuting_with_pauli) > 1:
            for i in range(1, len(generators_anticommuting_with_pauli)):
                new[generators_anticommuting_with_pauli[i]] = new[generators_anticommuting_with_pauli[i]].dot(new[generators_anticommuting_with_pauli[0]])
        # remove the first generator that anticommutes with the operator being measured
        generators_commuting_with_pauli = new.commutes_with_all(pauli)
        new = StabilizerGroup(new[generators_commuting_with_pauli])
        return new

    @property
    def x_z(self):
        return np.concatenate((self.x, self.z), axis=1)

    def copy(self):
        new = StabilizerGroup(super().copy())
        new.measurement_indexes = self.measurement_indexes.copy()
        return new

    def reduced(self, reversed=False):
        """Reduce the number of generators to the minimum number of independent generators, by gausian elimination.
        The index pivot keeps track of the row that is being used as a pivot.
        Goes column by column and multiplies all rows with non-zero element in that column by the first row with a
        non-zero element in that column which is below the pivot.
        Then moves the first row with a non-zero element in that column to the top.
        """
        pivot = 0
        new = StabilizerGroup(self.copy())
        for i in list(range(2*new.num_qubits))[::(-1 if reversed else 1)]:
            for j in range(pivot, len(new)):
                if new.x_z[j][i] != 0:
                    temp_row = new[j].copy()
                    new[j] = new[pivot]
                    new[pivot] = temp_row
                    # if j == len(self) - 1:
                    #     continue
                    for jj in range(len(new)):
                        if jj == pivot:
                            continue
                        if new.x_z[jj][i] != 0:
                            new[jj] = new[jj].dot(new[pivot])
                    pivot += 1
                    break
        # check if any of the generators are the minus identity
        if np.any([p == Pauli('-'+'I'*new.num_qubits) for p in new]):
            raise QiskitError('The Pauli Stabilizer group is not valid.')
        # remove rows after the pivot
        return StabilizerGroup(new[:pivot])

    def reduced_greedy_weight(self, num_iterations=1):
        new = self.copy()
        for _ in range(num_iterations):
            for i in range(len(new)):
                for j in range(i):
                    weight_i = pauli_weight(new[i])
                    weight_j = pauli_weight(new[j])
                    if weight_i >= weight_j:
                        i_to_replace = i
                    else:
                        i_to_replace = j
                    product_pauli = new[i].dot(new[j])
                    if pauli_weight(product_pauli) < max(weight_i, weight_j):
                        new[i_to_replace] = product_pauli
        return StabilizerGroup(new)

    def reduced_greedy_other(self, other: Pauli, num_iterations: int = 1):
        new = other.copy()
        weight = pauli_weight(new)
        for _ in range(num_iterations):
            for pauli in self:
                product_pauli = new.dot(pauli)
                product_weight = pauli_weight(product_pauli)
                if product_weight < weight:
                    new = product_pauli
                    weight = product_weight
        return new

    def reduced_by_candidate_elements(self, candidate_paulis):
        candidates_in_stabilizer = [pauli for pauli in candidate_paulis if self.contains_pauli(pauli, ignore_sign=True)]
        new = StabilizerGroup(self.copy()[:0])
        for pauli in candidates_in_stabilizer:
            if not new.contains_pauli(pauli, ignore_sign=True):
                new = new.append(pauli)
        for pauli in self:
            if not new.contains_pauli(pauli, ignore_sign=True):
                new = new.append(pauli)
        return StabilizerGroup(new)

    def append(self, pauli):
        """Append a Pauli operator to the stabilizer group.

        Args:
            Pauli (Pauli): The Pauli operator to append.
        """
        return StabilizerGroup(self.insert(len(self), pauli))

    def is_abelian(self):
        """Check if the Pauli Stabilizer group is abelian.

        Returns:
            bool: Whether the Pauli Stabilizer group is abelian.
        """
        # check that each element commutes with all other elements
        for generator in self:
            if len(self.anticommutes_with_all(generator)) > 0:
                return False
        return True

    def is_hermitian(self):
        """Check if the Pauli Stabilizer group is Hermitian.

        Returns:
            bool: Whether the Pauli Stabilizer group is Hermitian.
        """
        # check that each element is Hermitian
        for generator in self:
            if not generator.phase in [0,2]:
                return False
        return True

    # def insert(self, ind: int, value: PauliList, qubit: bool = False) -> Union[PauliSubGroup, Stabilizer]:
    #     return self.__class__(super().insert(ind, value, qubit=False))

    def contains_pauli(self, pauli: Pauli, ignore_sign=True) -> bool:
        num_generators = self.reduced().shape[0]
        if not isinstance(pauli, Pauli):
            raise 'Invalid pauli'
        try:
            with_pauli = StabilizerGroup(self.insert(num_generators, pauli)).reduced()
        except QiskitError as error:
            if error.message == 'The Pauli Stabilizer group is not valid.':
                return ignore_sign
            raise error
        return with_pauli.shape[0] == num_generators

    @property
    def pauli_normalizer(self) -> StabilizerGroup:
        if self._pauli_normalizer is None:
            array_flipped_z_x = galois.GF2(np.concatenate([self.z, self.x], axis=1).T.astype(int))
            self._pauli_normalizer = StabilizerGroup(StabilizerState(array_flipped_z_x.left_null_space()))
        return self._pauli_normalizer

    @property
    def logical_Zs(self) -> StabilizerGroup:
        if self._logical_Zs is None:
            stabilizer_with_logical_Zs = StabilizerGroup(self.copy())
            for p in self.pauli_normalizer[::-1]: #start with those with a lot of Zs
                if len(stabilizer_with_logical_Zs.anticommutes_with_all(p)) == 0 and not stabilizer_with_logical_Zs.contains_pauli(p, ignore_sign=True):
                    stabilizer_with_logical_Zs = StabilizerGroup(stabilizer_with_logical_Zs.insert(stabilizer_with_logical_Zs.shape[0], p))
            self._logical_Zs = stabilizer_with_logical_Zs[self.shape[0]:]
        return self._logical_Zs

    @property
    def logical_Xs(self) -> StabilizerGroup:
        if self._logical_Xs is None:
            stabilizer_with_logical_Xs = self.copy()
            for Z in self.logical_Zs:
                for p in self.pauli_normalizer.reduced(reversed=True):
                    if (len(stabilizer_with_logical_Xs.anticommutes_with_all(p)) == 0
                            and not StabilizerGroup(stabilizer_with_logical_Xs).contains_pauli(p)
                            and Z.anticommutes(p)):
                        stabilizer_with_logical_Xs = stabilizer_with_logical_Xs.insert(stabilizer_with_logical_Xs.shape[0], p)
                        break
            self._logical_Xs = stabilizer_with_logical_Xs[self.shape[0]:]
            for i,Z in enumerate(self.logical_Zs):
                idxs = self._logical_Xs.anticommutes_with_all(Z).tolist()
                idxs.remove(i)
                self._logical_Xs[idxs] = self._logical_Xs[idxs].dot(self._logical_Xs[i])
        return self._logical_Xs

    def is_logical_operator(self, pauli:Pauli):
        return self.commutes(pauli).all() and not self.contains_pauli(pauli, ignore_sign=True)

