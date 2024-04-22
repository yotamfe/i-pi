"""Creates objects to deal with dynamics driven by external fields."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2015 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np

from ipi.utils.depend import *
from ipi.engine.motion.driven_dynamics import EDA
from ipi.utils.units import Constants
from ipi.engine.motion.dynamics import NVEIntegrator, DummyIntegrator, Dynamics
from ipi.utils.depend import *
from ipi.utils.units import UnitMap
import re

# __all__ = ["BEC", "ElectricField", "EDA"]


class DrivenDynamics(Dynamics):
    """self (path integral) molecular dynamics class.

    Gives the standard methods and attributes needed in all the
    dynamics classes.

    Attributes:
        beads: A beads object giving the atoms positions.
        cell: A cell object giving the system box.
        forces: A forces object giving the virial and the forces acting on
            each bead.
        prng: A random number generator object.
        nm: An object which does the normal modes transformation.

    Depend objects:
        econs: The conserved energy quantity appropriate to the given
            ensemble. Depends on the various energy terms which make it up,
            which are different depending on the ensemble.he
        temp: The system temperature.
        dt: The timestep for the algorithms.
        ntemp: The simulation temperature. Will be nbeads times higher than
            the system temperature as PIMD calculations are done at this
            effective classical temperature.
    """

    def __init__(
        self,
        efield=None,
        bec=None,
        *argc,
        **argv,
    ):
        """Initialises a "dynamics" motion object.

        Args:
            dt: The timestep of the simulation algorithms.
            fixcom: An optional boolean which decides whether the centre of mass
                motion will be constrained or not. Defaults to False.
        """

        super().__init__(*argc, **argv)

        if self.enstype == "eda-nve":
            # NVE integrator with an external time-dependent driving (electric field)
            self.integrator = EDANVEIntegrator()
        else:
            self.integrator = DummyIntegrator()

        # if the dynamics is driven, allocate necessary objects
        self.efield = efield
        self.bec = bec
        self.eda = EDA(self.efield, self.bec)

    def bind(self, ens, beads, nm, cell, bforce, prng, omaker):

        super().bind(ens, beads, nm, cell, bforce, prng, omaker)

        self.eda.bind(self.ensemble, self.enstype)

        # now that the timesteps are decided, we proceed to bind the integrator.
        self.integrator.bind(self)

        # applies constraints immediately after initialization.
        self.integrator.pconstraints()

    def step(self, step=None):
        """Advances the dynamics by one time step"""

        super().step(step)
        # self.integrator.step(step)
        # self.ensemble.time += self.dt  # increments internal time

        # Check that these variable are the same.
        # If they are not the same, then there is a bug in the code
        dt = abs(self.ensemble.time - self.integrator.efield_time)
        if dt > 1e-12:
            raise ValueError(
                "The time at which the Electric Field is evaluated is not properly updated!"
            )


dproperties(DrivenDynamics, ["dt", "nmts", "splitting", "ntemp"])


class EDAIntegrator(DummyIntegrator):
    """Integrator object for simulations using the Electric Dipole Approximation (EDA)
    when an external electric field is applied.
    """

    def bind(self, motion):
        """bind variables"""
        super().bind(motion)

        self._time = self.eda._time
        self._mts_time = self.eda._mts_time

        dep = [
            self._time,
            self._mts_time,
            self.eda.Born_Charges._bec,
            self.eda.Electric_Field._Efield,
        ]
        self._EDAforces = depend_array(
            name="EDAforces",
            func=self._eda_forces,
            value=np.zeros((self.beads.nbeads, self.beads.natoms * 3)),
            dependencies=dep,
        )
        pass

    def pstep(self, level=0):
        """Velocity Verlet momentum propagator."""
        self.beads.p += self.EDAforces * self.pdt[level]
        if dstrip(self.efield_time) == dstrip(self.time):
            # it's the first time that 'pstep' is called
            # then we need to update 'efield_time'
            self.efield_time += dstrip(self.dt)
            # the next time this condition will be 'False'
            # so we will avoid to re-compute the EDAforces
        pass

    def _eda_forces(self):
        """Compute the EDA contribution to the forces, i.e. `q_e Z^* @ E(t)`"""
        Z = dstrip(self.eda.Born_Charges.bec)  # tensor of shape (nbeads,3xNatoms,3)
        E = dstrip(self.eda.Electric_Field.Efield)  # vector of shape (3)
        forces = Constants.e * Z @ E  # array of shape (nbeads,3xNatoms)
        return forces

    def step(self, step=None):
        if len(self.nmts) > 1:
            raise ValueError(
                "EDAIntegrator is not implemented with the Multiple Time Step algorithm (yet)."
            )
        # This should call 'NVEIntegrator.step' since 'self' should be an instance of 'EDANVEIntegrator' and not of 'EDAIntegrator'
        super().step(step)


dproperties(EDAIntegrator, ["EDAforces", "efield_time", "time"])


class EDANVEIntegrator(EDAIntegrator, NVEIntegrator):
    """Integrator object for simulations with constant Number of particles, Volume, and Energy (NVE)
    using the Electric Dipole Approximation (EDA) when an external electric field is applied.
    """

    def pstep(self, level):
        # NVEIntegrator does not use 'super()' within 'pstep'
        # then we can not use 'super()' here.
        # We need to call the 'pstep' methods explicitly.
        NVEIntegrator.pstep(
            self, level
        )  # the driver is called here: add nuclear and electronic forces (DFT)
        EDAIntegrator.pstep(self, level)  # add the driving forces, i.e. q_e Z @ E(t)
        pass


class BEC:
    """Class to handle the Born Effective Charge tensors when performing driven dynamics (with 'eda-nve')"""

    # The BEC tensors Z^* are defined as the derivative of the electric dipole of the system w.r.t. nuclear positions
    # in units of the elementary charge, i.e. Z^*_{ij} = 1/q_e \frac{\partial d_i }{\partial R_j}
    # The dipole is a vector of shape (3), while the nuclear positions have shape (3xNatoms)
    # The BEC tensors have then shape (3xNatoms,3).
    # A tensor of this shape is stored for each bead --> self._bec.shape = (self.nbeads, 3 * self.natoms, 3)
    #
    # If an external time-dependent electric field E(t) is applied, this couples to the dipole of the system,
    # and the resulting additional term to the forces is given by q_e Z^* @ E(t) --> have a look at EDAIntegrator._eda_forces
    #
    # The BEC tensors Z^* can be given to i-PI by an external driven through the etxra strings in the forces
    # of they can be kept fixed during the dynamics: in this case you can provide them through a txt file.

    def __init__(self, cbec=None, bec=None):
        self.cbec = cbec
        if bec is None:
            bec = np.full((0, 3), np.nan)
        self._bec = depend_array(name="bec", value=bec)
        pass

    def bind(self, eda, ensemble, enstype):
        self.enstype = enstype
        self.nbeads = ensemble.beads.nbeads
        self.natoms = ensemble.beads.natoms
        self.forces = ensemble.forces

        if self.enstype in EDA.integrators and self.cbec:
            self._bec = depend_array(
                name="bec",
                value=np.full((self.nbeads, 3 * self.natoms, 3), np.nan),
                func=self._get_driver_BEC,
                dependencies=[ensemble.beads._q],
            )
        elif self.enstype in EDA.integrators:
            temp = self._get_fixed_BEC()  # reshape the BEC once and for all
            self._bec = depend_array(name="bec", value=temp)
        else:
            self._bec = depend_array(
                name="bec", value=np.full((self.nbeads, 3 * self.natoms, 3), np.nan)
            )

        pass

    def store(self, bec):
        super(BEC, self).store(bec)
        self.cbec.store(bec.cbec)

    def _get_driver_BEC(self, bead=None):
        """Return the BEC tensors (in cartesian coordinates), when computed by the driver"""

        msg = "Error in '_get_driver_BEC'"

        if bead is not None:
            if bead < 0:
                raise ValueError("Error in '_get_driver_BEC': 'bead' is negative")
            if bead >= self.nbeads:
                raise ValueError(
                    "Error in '_get_driver_BEC': 'bead' is greater than the number of beads"
                )
        else:
            if self.nbeads != 1:
                raise ValueError(
                    "Error in '_get_driver_BEC': EDA integration has not implemented yet for 'nbeads' > 1"
                )

        if self.cbec:
            if "BEC" not in self.forces.extras:
                raise ValueError(
                    msg
                    + ": BEC tensors are not returned to i-PI (or at least not accessible in '_get_driver_BEC')."
                )
        else:
            raise ValueError(
                msg + ": you should not get into this functon if 'cbec' is False."
            )

        BEC = np.full((self.nbeads, 3 * self.natoms, 3), np.nan)
        for n in range(self.nbeads):
            bec = np.asarray(self.forces.extras["BEC"][n])

            if bec.shape[0] != 3 * self.natoms:
                raise ValueError(
                    msg
                    + ": number of BEC tensors is not equal to the number fo atoms x 3."
                )
            if bec.shape[1] != 3:
                raise ValueError(
                    msg
                    + ": BEC tensors with wrong shape. They should have 3 components."
                )

            BEC[n, :, :] = np.copy(bec)

        return BEC

    def _get_fixed_BEC(self):
        """Return the BEC tensors (in cartesian coordinates).
        The BEC tensor are stored in a compact form.
        This method trasform the BEC tensors into another data structure, suitable for computation.
        A lambda function is also returned to perform fast matrix multiplication.
        """
        try:
            return self.bec.reshape((self.nbeads, 3 * self.natoms, 3))
        except:
            line = (
                "Error in '_get_fixed_BEC': i-PI is going to stop.\n"
                + "The BEC tensor is: "
                + str(self.bec)
                + "\nPay attention that in 'input.xml' you should have the following line:\n"
                + "\t'<bec mode=\"file\"> filepath </bec>'\n"
                + "The default mode could be 'none'. Please change it."
            )
            raise ValueError(line)


dproperties(BEC, ["bec"])


class ElectricDipole:
    """Class to handle the electric dipole of the system when performing driven dynamics (with 'eda-nve')"""

    def __init__(self):
        pass

    def bind(self, eda, ensemble):
        self._nbeads = depend_value(name="nbeads", value=ensemble.beads.nbeads)

        self.ens = ensemble

        val = np.full((self.nbeads, 3), np.nan)
        self._dipole = depend_array(
            name="dipole",
            func=lambda: self._get_dipole(),
            value=val,
            dependencies=[ensemble.beads._q],
        )

        pass

    def store(self, dipole):
        super().store(dipole)
        pass

    def _get_dipole(self, bead=None):
        """Return the electric dipole of all the beads as a list of np.array"""

        # check that 'bead' is correct
        if bead is not None:
            if bead < 0:
                raise ValueError("Error in '_get_dipole': 'beads' is negative.")
            if bead >= self.nbeads:
                raise ValueError(
                    "Error in '_get_dipole': 'beads' is greater than the number of beads."
                )

        dipole = np.full((self.nbeads, 3), np.nan)
        try:
            if "dipole" in self.ens.forces.extras:
                raws = [self.ens.forces.extras["dipole"][i] for i in range(self.nbeads)]
                for n, raw in enumerate(raws):
                    if len(raw) != 3:
                        raise ValueError("'dipole' has not length 3")
                    dipole[n] = np.asarray(raw)
                return dipole

            elif "raw" not in self.ens.forces.extras:
                raise ValueError("'raw' has to be in 'forces.extras'")

            elif np.all(
                ["Total dipole moment" in s for s in self.ens.forces.extras["raw"]]
            ):
                raws = [self.ens.forces.extras["raw"][i] for i in range(self.nbeads)]
                for n, raw in enumerate(raws):
                    factor = 1.0
                    if "[eAng]" in raw:
                        factor = UnitMap["length"]["angstrom"]

                    pattern = r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\b[-+]?\d+\b"
                    matches = list(re.findall(pattern, raw))
                    if len(matches) != 3:
                        raise ValueError(
                            "wrong number of extracted values from the extra string: they should be 3."
                        )
                    else:
                        dipole[n] = float(factor) * np.asarray(matches)
                    return dipole
            else:
                raise ValueError(
                    "Error in '_get_dipole': can not extract dipole from the extra string."
                )
        except:
            return np.full((self.nbeads, 3), np.nan)


dproperties(ElectricDipole, ["dipole", "nbeads", "forces"])


class ElectricField:
    """Class to handle the time dependent electric field when performing driven dynamics (with 'eda-nve')"""

    def __init__(self, amp=None, freq=None, phase=None, peak=None, sigma=None):
        self._amp = depend_array(
            name="amp", value=amp if amp is not None else np.zeros(3)
        )
        self._freq = depend_value(name="freq", value=freq if freq is not None else 0.0)
        self._phase = depend_value(
            name="phase", value=phase if phase is not None else 0.0
        )
        self._peak = depend_value(name="peak", value=peak if peak is not None else 0.0)
        self._sigma = depend_value(
            name="sigma", value=sigma if sigma is not None else np.inf
        )
        self._Efield = depend_array(
            name="Efield",
            value=np.zeros(3, float),
            func=lambda: np.zeros(3, float),
        )
        self._Eenvelope = depend_value(
            name="Eenvelope",
            value=1.0,
            func=lambda: np.zeros(3, float),
        )

    def bind(self, eda, enstype):
        self.enstype = enstype
        self._mts_time = eda._mts_time

        dep = [self._mts_time, self._peak, self._sigma]
        self._Eenvelope = depend_value(
            name="Eenvelope", value=1.0, func=self._get_Eenvelope, dependencies=dep
        )

        if enstype in EDA.integrators:
            # dynamics is not driven --> add dependencies to the electric field
            dep = [self._mts_time, self._amp, self._freq, self._phase, self._Eenvelope]
            self._Efield = depend_array(
                name="Efield",
                value=np.zeros(3, float),
                func=self._get_Efield,
                dependencies=dep,
            )
        else:
            # dynamics is not driven --> no dependencies for the electric field
            self._Efield = depend_array(
                name="Efield",
                value=np.zeros(3, float),
                func=lambda: np.zeros(3, float),
            )
        pass

    def store(self, ef):
        super(ElectricField, self).store(ef)
        self.amp.store(ef.amp)
        self.freq.store(ef.freq)
        self.phase.store(ef.phase)
        self.peak.store(ef.peak)
        self.sigma.store(ef.sigma)
        pass

    def _get_Efield(self):
        """Get the value of the external electric field (cartesian axes)"""
        time = dstrip(self.efield_time)
        if hasattr(time, "__len__"):
            return np.outer(self._get_Ecos(time) * self.Eenvelope, self.amp)
        else:
            return self._get_Ecos(time) * self.Eenvelope * self.amp

    def _Eenvelope_is_on(self):
        return self.peak > 0.0 and self.sigma != np.inf

    def _get_Eenvelope(self):
        time = dstrip(self.efield_time)
        """Get the gaussian envelope function of the external electric field"""
        if self._Eenvelope_is_on():
            x = time  # indipendent variable
            u = self.peak  # mean value
            s = self.sigma  # standard deviation
            return np.exp(
                -0.5 * ((x - u) / s) ** 2
            )  # the returned maximum value is 1, when x = u
        else:
            return 1.0

    def _get_Ecos(self, time):
        """Get the sinusoidal part of the external electric field"""
        return np.cos(self.freq * time + self.phase)


dproperties(
    ElectricField,
    ["amp", "phase", "peak", "sigma", "freq", "Eenvelope", "Efield", "efield_time"],
)


class EDA:
    """Class to handle in a compact way 'BEC', 'ElectricDipole', and 'ElectricField' objects when performing driven dynamics (with 'eda-nve')"""

    integrators = ["eda-nve"]

    def __init__(self, efield: ElectricField, bec: BEC, **kwargv):
        super(EDA, self).__init__(**kwargv)
        self.Electric_Field = efield
        self.Electric_Dipole = ElectricDipole()
        self.Born_Charges = bec
        self._time = depend_value(name="time", value=0)
        self._mts_time = depend_value(name="efield_time", value=0)
        pass

    def bind(self, ensemble, enstype):
        self.enstype = enstype
        self._mts_time = depend_value(name="efield_time", value=dstrip(ensemble).time)
        self._time = ensemble._time
        self.Electric_Field.bind(self, enstype)
        self.Electric_Dipole.bind(self, ensemble)
        self.Born_Charges.bind(self, ensemble, enstype)
        pass

    def store(self, eda):
        super(EDA, self).store(eda)
        self.Electric_Field.store(eda.Electric_Field)
        self.Electric_Dipole.store(eda.Electric_Dipole)
        self.Born_Charges.store(eda.bec)
        pass


dproperties(EDA, ["efield_time", "time"])
