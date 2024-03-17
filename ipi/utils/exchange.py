"""Contains all methods to evalaute potential energy and forces for indistinguishable particles.
Used in /engine/normalmodes.py
"""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import math
import numpy as np
from ipi.utils.depend import *
import ipi.utils.units as units
import sys

def kth_diag_indices(a, k):
    """
    Indices to access matrix k-diagonals in numpy.
    https://stackoverflow.com/questions/10925671/numpy-k-th-diagonal-indices
    """
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


class ExchangePotential:
    def __init__(self, nbeads, nbosons):
        assert nbosons > 0
        self.nbosons = nbosons
        self.nbeads = nbeads

    def bind(self, beads, ensemble, nm):
        self.beads = beads
        self.ensemble = ensemble

        self._omegan2 = depend_value(name="omegan2", value=0.0)
        dpipe(nm._omegan2, self._omegan2)

        self._bosons = depend_array(
            name="bosons", value=np.zeros(self.nbosons, dtype=int)
        )
        dpipe(nm._bosons, self._bosons)

        self._betaP = depend_value(
            "betaP",
            func=lambda: 1.0
            / (self.beads.nbeads * units.Constants.kb * self.ensemble.temp),
            dependencies=[self.ensemble._temp],
        )

        # the boson mass is assumed to be that of the first particle
        self._boson_m = depend_value(
            name="boson_m",
            func=self.get_boson_mass,
            dependencies=[self.beads._m, self._bosons],
        )

        # boson coordinates are just a slice of the beads coordinates
        self._qbosons = depend_array(
            name="qbosons",
            value=np.zeros((self.nbeads, self.nbosons, 3)),
            func=lambda: dstrip(self.beads.q).reshape(self.nbeads, -1, 3)[
                :, self.bosons
            ],
            dependencies=[self.beads._q, self.bosons],
        )

        # self.bead_dist_inter_first_last_bead[j][l][m] = r^{j+1}_{l} - r^{j}_{m}
        self._bead_diff_inter_first_last_bead = depend_array(
            name="bead_diff_inter_first_last_bead",
            value=np.zeros((self.nbeads, self.nbosons, self.nbosons, 3)),
            func=lambda: (
                np.roll(dstrip(self.qbosons), axis=0, shift=-1)[:, :, np.newaxis, :]
                - dstrip(self.qbosons)[:, np.newaxis, :, :]
            ),
            dependencies=[self._qbosons],
        )
        # cycle energies:
        # self.cycle_energies[j, u, v] is the contribution of imaginary time slice j
        # in a permutation where there is a cycle on particle indices u,...,v
        self._cycle_energies = depend_array(
            name="cycle_energies",
            value=np.zeros((self.nbeads, self.nbosons, self.nbosons), dtype=float, order="F"),
            func=self.get_cycle_energies,
            dependencies=[
                self._boson_m,
                self._omegan2,
                self._bead_diff_inter_first_last_bead,
            ],
        )
        # prefix potentials:
        # self.prefix_V[l] = V^[1,l+1]
        self._prefix_V = depend_array(
            name="V",
            func=self.get_prefix_V,
            value=np.zeros((self.nbeads, self.nbosons + 1), float),
            dependencies=[self._cycle_energies, self._betaP],
        )

        # suffix potentials:
        # self.suffix_V[l] = V^[l+1, N]
        self._suffix_V = depend_value(
            name="VB",
            func=self.get_suffix_V,
            value=np.zeros((self.nbeads, self.nbosons + 1), float),
            dependencies=[self._cycle_energies, self._betaP],
        )

        # this is the main quantity used outside this
        self._vspring_and_fspring = depend_value(
            name="vspring_and_fspring",
            func=self.get_vspring_and_fspring,
            dependencies=[
                self._betaP,
                self._omegan2,
                self._prefix_V,
                self._suffix_V,
                self._cycle_energies,
                self._bead_diff_inter_first_last_bead,
            ],
        )

        # properties
        self._kinetic_td = depend_value(
            name="kinetic_td",
            func=self.get_kinetic_td,
            dependencies=[self._betaP, self._prefix_V, self._cycle_energies],
        )

        self._fermionic_sign = depend_value(
            name="fermionic_sign",
            func=self.get_fermionic_sign,
            dependencies=[self._betaP, self._prefix_V, self._cycle_energies],
        )

        self._longest_probability = depend_value(
            name="longest_probability",
            func=self.get_longest_probability,
            dependencies=[self._betaP, self._prefix_V, self._cycle_energies],
        )

        self._distinct_probability = depend_value(
            name="distinct_probability",
            func=self.get_distinct_probability,
            dependencies=[self._betaP, self._prefix_V, self._cycle_energies],
        )

    def get_boson_mass(self):
        """
        Ensures that all the bosons have the same mass, and returns it.
        """
        masses = dstrip(self.beads.m)[self.bosons]
        if len(set(masses)) > 1:
            raise ValueError(
                "Bosons must have the same mass, found %s for bosons %s"
                % (str(masses), str(self.bosons))
            )
        return masses[0]

    def get_cycle_energies(self):
        """
        Evaluate all the cycle energies, as outlined in Eqs. 5-7 of arXiv:2305.18025.
        Returns an upper-triangular matrix, Emks[u,v] is the ring polymer energy of the cycle on u,...,v.
        """
        # using column-major (Fortran order) because uses of the array are mostly within the same column
        Emks = np.zeros((self.nbeads, self.nbosons, self.nbosons), dtype=float, order="F")

        spring_energy_first_last_bead_array = np.sum(
            dstrip(self.bead_diff_inter_first_last_bead) ** 2, axis=-1
        )

        spring_potential_prefix = 0.5 * self.boson_m * self.omegan2

        # for m in range(self.nbosons):
        #     Emks[j][m][m] = spring_potential_prefix * (spring_energy_first_last_bead_array[j, m, m])
        for j in range(self.nbeads):
            Emks[j, np.diag_indices_from(Emks[j])] = spring_potential_prefix * (
                np.diagonal(spring_energy_first_last_bead_array[j])
        )

        for s in range(self.nbosons - 1 - 1, -1, -1):
            # for m in range(s + 1, self.nbosons):
            #     Emks[j][s][m] = Emks[j][s + 1][m] + spring_potential_prefix * (
            #             - spring_energy_first_last_bead_array[j][s + 1, m]
            #             + spring_energy_first_last_bead_array[j][s + 1, s]
            #             + spring_energy_first_last_bead_array[j][s, m])
            Emks[:, s, (s + 1) :] = Emks[:, s + 1, (s + 1) :] + spring_potential_prefix * (
                -spring_energy_first_last_bead_array[:, s + 1, (s + 1) :]
                + spring_energy_first_last_bead_array[:, s + 1, s, np.newaxis]
                + spring_energy_first_last_bead_array[:, s, (s + 1) :]
            )

        return Emks

    def get_prefix_V(self):
        """
        Evaluate V^[1,1], V^[1,2], ..., V^[1,N], as outlined in Eq. 3 of arXiv:2305.18025.
        (In the code, particle indices start from 0.)
        Returns a vector of these potentials, in this order.
        Assumes that the cycle energies self.cycle_energies have been computed.
        """
        V = np.zeros((self.nbeads, self.nbosons + 1), float)

        for m in range(1, self.nbosons + 1):
            # For numerical stability
            subdivision_potentials = V[:, :m] + dstrip(self.cycle_energies)[:, :m, m - 1]
            Elong = np.min(subdivision_potentials, axis=-1)

            # sig = 0.0
            # for u in range(m):
            #   sig += np.exp(- self.betaP *
            #                (V[u] + self.cycle_energies[u, m - 1] - Elong) # V until u-1, then cycle from u to m
            #                 )
            sig = np.sum(np.exp(-self.betaP * (subdivision_potentials - Elong[:, np.newaxis])), axis=-1)
            assert sig.all() and np.all(np.isfinite(sig))
            V[:, m] = Elong - np.log(sig / m) / self.betaP

        return V

    def get_suffix_V(self):
        """
        Evaluate V^[1,N], V^[2,N], ..., V^[N,N], as outlined in Eq. 16 of arXiv:2305.18025.
        (In the code, particle indices start from 0.)
        Returns a vector of these potentials, in this order.
        Assumes that both the cycle energies self.cycle_energies and prefix potentials self.prefix_V have been computed.
        """
        RV = np.zeros((self.nbeads, self.nbosons + 1), float)

        for l in range(self.nbosons - 1, 0, -1):
            # For numerical stability
            subdivision_potentials = dstrip(self.cycle_energies)[:, l, l:] + RV[:, l + 1 :]
            Elong = np.min(subdivision_potentials, axis=-1)

            # sig = 0.0
            # for p in range(l, self.nbosons):
            #     sig += 1 / (p + 1) * np.exp(- self.betaP * (self.cycle_energies[l, p] + RV[p + 1]
            #                                                 - Elong))
            sig = np.sum(
                np.reciprocal(np.arange(l + 1.0, self.nbosons + 1))
                * np.exp(-self.betaP * (subdivision_potentials - Elong[:, np.newaxis])),
                axis=-1
            )
            assert sig.all() and np.all(np.isfinite(sig))
            RV[:, l] = Elong - np.log(sig) / self.betaP

        # V^[1,N]
        RV[:, 0] = self.prefix_V[:, -1]

        return RV

    def _connection_probabilities(self, j):
        connection_probs = np.zeros((self.nbosons, self.nbosons), float)
        # close cycle probabilities:
        # TODO:
        # for u in range(0, self._N):
        #     for l in range(u, self._N):
        #         connection_probs[l][u] = 1 / (l + 1) * \
        #                np.exp(- self._betaP *
        #                        (self._V[u] + self._E_from_to[u, l] + self._V_backward[l+1]
        #                         - self.V_all()))
        tril_indices = np.tril_indices(self.nbosons, k=0)
        connection_probs[tril_indices] = (
            # np.asarray([1 / (l + 1) for l in range(self._N)])[:, np.newaxis] *
                np.reciprocal(np.arange(1.0, self.nbosons + 1))[:, np.newaxis]
                * np.exp(
            -self.betaP
            * (
                # np.asarray([self._V(u - 1) for u in range(self._N)])[np.newaxis, :]
                    self.prefix_V[j][np.newaxis, :-1]
                    # + np.asarray([(self._E_from_to[u, l] if l >= u else 0) for l in range(self._N)
                    #                   for u in range(self._N)]).reshape((self._N, self._N))
                    + self.cycle_energies[j].T
                    # + np.asarray([self._V_backward(l + 1) for l in range(self._N)])[:, np.newaxis]
                    + self.suffix_V[j][1:, np.newaxis]
                    - self.prefix_V[j][self.nbosons]
            )
        )
        )[tril_indices]
        # direct link probabilities:
        # TODO:
        # for l in range(self._N - 1):
        #     connection_probs[l][l+1] = 1 - (np.exp(- self._betaP * (self._V[l + 1] + self._V_backward[l + 1] -
        #                                         self.V_all())))
        superdiagonal_indices = kth_diag_indices(connection_probs, k=1)
        connection_probs[superdiagonal_indices] = 1 - (
            np.exp(
                -self.betaP * (self.prefix_V[j][1:-1] + self.suffix_V[j][1:-1] - self.prefix_V[j][-1])
            )
        )
        return connection_probs

    def get_vspring_and_fspring(self):
        """
        Returns spring potential and forces for bosons, as a tuple (V, F).
        The potential is obtained from _prefix_V[N].
        Evaluate the ring polymer forces on all the beads, as outlined in Eq. 13, 17-18 of arXiv:2305.18025.
        (In the code, particle indices start from 0.)
        In the array, F[j, l, :] is the force (3d vector) on bead j of particle l.
        Assumes that both the cycle energies self.cycle_energies, the prefix potentials self.prefix_V,
        and the suffix potentials self.suffix_V have been computed.
        """
        F = np.zeros((self.nbeads, self.nbosons, 3), float)

        connection_probs = np.empty((self.nbeads, self.nbosons, self.nbosons))
        for j in range(self.nbeads):
            connection_probs[j] = self._connection_probabilities(j)

        # # workaround - no exchange except one location
        # for j in range(self._P - 1):
        #     connection_probs[j] = np.zeros(self._N)
        #     for l in range(self._N):
        #         connection_probs[j, l, l] = 1.0

        # bead_diff_intra = np.diff(self._q, axis=0)  # TODO: remove

        spring_force_prefix = (-1.0) * self.boson_m * self.omegan2

        for j in range(self.nbeads):
            # if j not in [0, self._P - 1]:
            #     F[j, :, :] = self._spring_force_prefix() * (-bead_diff_intra[j, :] + bead_diff_intra[j - 1, :])
            #     continue

            # TODO: doc
            # on the last bead:
            #
            # for l in range(self._N):
            #     force_from_neighbor = np.empty((self._N, 3))
            #     for next_l in range(max(l + 1, self._N)):
            #         force_from_neighbor[next_l, :] = self._spring_force_prefix() * \
            #                         (-self._bead_diff_inter_first_last_bead[next_l, l] + self._bead_diff_intra[-1, l])
            #     F[-1, l, :] = sum(connection_probs[l][next_l] * force_from_neighbor[next_l]
            #                       for next_l in range(self._N))
            #
            # First vectorization:
            # for l in range(self._N):
            #     force_from_neighbors = np.empty((self._N, 3))
            #     force_from_neighbors[:, :] = self._spring_force_prefix() * \
            #                         (-self._bead_diff_inter_first_last_bead[:, l] + self._bead_diff_intra[-1, l])
            #     F[-1, l, :] = np.dot(connection_probs[l][:], force_from_neighbors)
            force_from_next_neighbor = spring_force_prefix * (
                -np.transpose(self._bead_diff_inter_first_last_bead[j], axes=(1, 0, 2))
            )
            weighted_force_from_next_slice = np.einsum("ljk,lj->lk", force_from_next_neighbor, connection_probs[j])
            force_from_prev_neighbor = spring_force_prefix * (
                self._bead_diff_inter_first_last_bead[j - 1]
            )
            weighted_force_from_prev_slice = np.einsum("ljk,jl->lk", force_from_prev_neighbor, connection_probs[j - 1])
            # TODO: doc
            # F[-1, l, k] = sum_{j}{force_from_neighbors[l][j][k] * connection_probs[l,j]}
            F[j, :, :] = weighted_force_from_next_slice + weighted_force_from_prev_slice

        # np.set_printoptions(precision=8, suppress=False)
        # print("round")
        # print("probabilities")
        # print(connection_probs[-1])
        # print("forces")
        # print(F)

        return [self.logsumboltz(self.prefix_V[:, -1]), F]

    def logsumboltz(self, arr):
        Elong = np.min(arr)
        sig = np.mean(np.exp(-self.betaP * (arr - Elong)))
        assert sig.all() and np.all(np.isfinite(sig))
        return Elong - np.log(sig) / self.betaP

    def get_distinct_probability(self):
        """
        Evaluate the probability of the configuration where all the particles are separate.
        """
        return np.exp(
            -self.betaP
            * (
                np.trace(dstrip(self.cycle_energies))
                - dstrip(self.prefix_V)[self.nbosons]
            )
            - math.log(
                np.math.factorial(self.nbosons)
            )  # (1.0 / np.math.factorial(self.nbosons))
        )

    def get_longest_probability(self):
        """
        Evaluate the probability of a configuration where all the particles are connected,
        divided by 1/N. Notice that there are (N-1)! permutations of this topology
        (all represented by the cycle 0,1,...,N-1,0); this cancels the division by 1/N.
        """
        return np.exp(
            -self.betaP
            * (dstrip(self.cycle_energies)[0, -1] - dstrip(self.prefix_V)[self.nbosons])
        )

    def get_kinetic_td(self):
        """Implementation of the Hirshberg-Rizzi-Parrinello primitive
        kinetic energy estimator for identical particles.
        Corresponds to Eqns. (4)-(5) in SI of pnas.1913365116.
        """
        # TODO:
        return 0.0

        cycle_energies = dstrip(self.cycle_energies)
        prefix_V = dstrip(self.prefix_V)

        est = np.zeros(self.nbosons + 1)

        for m in range(1, self.nbosons + 1):
            sig = 0.0

            # Numerical stability - Xiong-Xiong method (arXiv.2206.08341)
            e_tilde = sys.float_info.max
            for k in range(m, 0, -1):
                e_tilde = min(e_tilde, cycle_energies[m - k, m - 1] + prefix_V[m - k])

            for k in range(m, 0, -1):
                E_kn_val = cycle_energies[m - k, m - 1]
                sig += (est[m - k] - E_kn_val) * np.exp(
                    -self.betaP * (E_kn_val + prefix_V[m - k] - e_tilde)
                )

            sig_denom_m = m * np.exp(-self.betaP * (prefix_V[m] - e_tilde))

            est[m] = sig / sig_denom_m

        factor = 1.5 * self.nbosons / self.betaP

        return factor + est[self.nbosons] / self.nbeads

    def get_fermionic_sign(self):
        """
        The average permutation sign as defined in Eq. (9) https://doi.org/10.1063/5.0008720,
        which can be used to reweight observables to obtain fermionic statistics.
        """
        # TODO:
        return 0.0

        return self._get_fermionic_potential_exp() / np.exp(
            -self.betaP * self.prefix_V[-1]
        )

    def _get_fermionic_potential_exp(self):
        """
        Exponential of the fermionic pseudo-potential defined by
        the recurrence relation in Eq. (5) of https://doi.org/10.1063/5.0008720.
        Numerically unstable since it does not use log-sum-exp trick, seeing that the
        sum of exponentials could be negative.
        """
        xi = -1
        W = np.empty(self.nbosons + 1, float)
        W[0] = 1.0

        for m in range(1, self.nbosons + 1):
            perm_sign = np.array([xi ** (k - 1) for k in range(m, 0, -1)])
            W[m] = (1.0 / m) * np.sum(
                perm_sign
                * W[:m]
                * np.exp(-self.betaP * dstrip(self.cycle_energies)[:m, m - 1])
            )

        return W[-1]


dproperties(
    ExchangePotential,
    [
        "omegan2",
        "bosons",
        "qbosons",
        "betaP",
        "boson_m",
        "bead_diff_inter_first_last_bead",
        "cycle_energies",
        "prefix_V",
        "suffix_V",
        "vspring_and_fspring",
        "kinetic_td",
        "distinct_probability",
        "longest_probability",
        "fermionic_sign",
    ],
)