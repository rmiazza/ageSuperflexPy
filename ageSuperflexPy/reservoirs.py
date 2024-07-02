from ageSuperflexPy.element import ODEsElement
import numba as nb
import numpy as np


class PowerReservoirHBV(ODEsElement):
    """
    This class implements the PowerReservoir present in HBV.
    """

    def __init__(self, parameters, states, approximation, id):
        """
        This is the initializer of the class PowerReservoir.

        Parameters
        ----------
        parameters : dict
            Parameters of the element. The keys must be:
            - 'k' : multiplier of the state
            - 'alpha' : exponent of the state
        states : dict
            Initial state of the element. The keys must be:
            - 'S0' : initial storage of the reservoir.
        approximation : superflexpy.utils.numerical_approximation.NumericalApproximator
            Numerial method used to approximate the differential equation
        id : str
            Itentifier of the element. All the elements of the framework must
            have an id.
        """

        ODEsElement.__init__(self,
                             parameters=parameters,
                             states=states,
                             approximation=approximation,
                             id=id)

        self._fluxes_python = [self._fluxes_function_python]  # Used by get fluxes, regardless of the architecture

        if approximation.architecture == 'numba':
            self._fluxes = [self._fluxes_function_numba]
        elif approximation.architecture == 'python':
            self._fluxes = [self._fluxes_function_python]

    # METHODS FOR THE USER

    def set_input(self, input):
        """
        Set the input of the element.

        Parameters
        ----------
        input : list(numpy.ndarray)
            List containing the input fluxes of the element. It contains 1
            flux:
            1. Rainfall
        """

        self.input = {'P': input[0]}

    def get_output(self, solve=True):
        """
        This method solves the differential equation governing the routing
        store.

        Returns
        -------
        list(numpy.ndarray)
            Output fluxes in the following order:
            1. Streamflow (Q)
        """

        if solve:
            self._solver_states = [self._states[self._prefix_states + 'S0']]
            self._solve_differential_equation()

            # Update the state
            self.set_states({self._prefix_states + 'S0': self.state_array[-1, 0]})

        fluxes = self._num_app.get_fluxes(fluxes=self._fluxes_python,  # I can use the python method since it is fast
                                          S=self.state_array,
                                          S0=self._solver_states,
                                          dt=self._dt,
                                          **self.input,
                                          **{k[len(self._prefix_parameters):]: self._parameters[k] for k in self._parameters},
                                          )

        return [- fluxes[0][1]]

    # PROTECTED METHODS

    @staticmethod
    def _fluxes_function_python(S, S0, ind, P, k, alpha, dt):

        if ind is None:
            return (
                [
                    P,
                    - k * S**alpha,
                ],
                0.0,
                S0 + P * dt
            )
        else:
            return (
                [
                    P[ind],
                    - k[ind] * S**alpha[ind],
                ],
                0.0,
                S0 + P[ind] * dt[ind],
                [
                    0.0,
                    - k[ind] * alpha[ind] * S**(alpha[ind] - 1)
                ]
            )

    @staticmethod
    @nb.jit('Tuple((UniTuple(f8, 2), f8, f8, UniTuple(f8, 2)))(optional(f8), f8, i4, f8[:], f8[:], f8[:], f8[:])',
            nopython=True)
    def _fluxes_function_numba(S, S0, ind, P, k, alpha, dt):
        # This method is used only when solving the equation

        return (
            (
                P[ind],
                - k[ind] * S**alpha[ind],
            ),
            0.0,
            S0 + P[ind] * dt[ind],
            (
                0.0,
                - k[ind] * alpha[ind] * S**(alpha[ind] - 1)
            )
        )


# Very slow for some unknown reason, probably due to the storage equation
class PowerReservoirNorm(ODEsElement):
    """
    This class implements the PowerReservoir presented in Kirchner (2016) - Aggregation in environmental systems – Part 2.
    """

    def __init__(self, parameters, states, approximation, id):
        """
        This is the initializer of the class PowerReservoir.

        Parameters
        ----------
        parameters : dict
            Parameters of the element. The keys must be:
            - 'I_bar' : long-term average input rate
            - 'S_ref' : reference storage value
            - 'b' : exponent of the normalized state
        states : dict
            Initial state of the element. The keys must be:
            - 'S0' : initial storage of the reservoir.
        approximation : superflexpy.utils.numerical_approximation.NumericalApproximator
            Numerial method used to approximate the differential equation
        id : str
            Itentifier of the element. All the elements of the framework must
            have an id.
        """

        ODEsElement.__init__(self,
                             parameters=parameters,
                             states=states,
                             approximation=approximation,
                             id=id)

        self._fluxes_python = [self._fluxes_function_python]  # Used by get fluxes, regardless of the architecture

        if approximation.architecture == 'numba':
            self._fluxes = [self._fluxes_function_numba]
        elif approximation.architecture == 'python':
            self._fluxes = [self._fluxes_function_python]

    # METHODS FOR THE USER

    def set_input(self, input):
        """
        Set the input of the element.

        Parameters
        ----------
        input : list(numpy.ndarray)
            List containing the input fluxes of the element. It contains 1
            flux:
            1. Rainfall
        """

        self.input = {'P': input[0]}

    def get_output(self, solve=True):
        """
        This method solves the differential equation governing the routing
        store.

        Returns
        -------
        list(numpy.ndarray)
            Output fluxes in the following order:
            1. Streamflow (Q)
        """

        if solve:
            self._solver_states = [self._states[self._prefix_states + 'S0']]
            self._solve_differential_equation()

            # Update the state
            self.set_states({self._prefix_states + 'S0': self.state_array[-1, 0]})

        fluxes = self._num_app.get_fluxes(fluxes=self._fluxes_python,  # I can use the python method since it is fast
                                          S=self.state_array,
                                          S0=self._solver_states,
                                          dt=self._dt,
                                          **self.input,
                                          **{k[len(self._prefix_parameters):]: self._parameters[k] for k in self._parameters},
                                          )

        return [- fluxes[0][1]]

    # PROTECTED METHODS

    @staticmethod
    def _fluxes_function_python(S, S0, ind, P, I_bar, S_ref, b, dt):

        if ind is None:
            return (
                [
                    P,
                    - I_bar * (S / S_ref)**b,
                ],
                0.0,
                S0 + P * dt
            )
        else:
            return (
                [
                    P[ind],
                    - I_bar[ind] * (S / S_ref[ind])**b[ind],
                ],
                0.0,
                S0 + P[ind] * dt[ind],
                [
                    0.0,
                    - I_bar[ind] * b[ind] / S_ref**b[ind] * S**(b[ind] - 1)
                ]
            )

    @staticmethod
    @nb.jit('Tuple((UniTuple(f8, 2), f8, f8, UniTuple(f8, 2)))(optional(f8), f8, i4, f8[:], f8[:], f8[:], f8[:], f8[:])',
            nopython=True)
    def _fluxes_function_numba(S, S0, ind, P, I_bar, S_ref, b, dt):
        # This method is used only when solving the equation

        return (
            (
                P[ind],
                - I_bar[ind] * (S / S_ref[ind])**b[ind],
            ),
            0.0,
            S0 + P[ind] * dt[ind],
            (
                0.0,
                - I_bar[ind] * b[ind] / S_ref[ind]**b[ind] * S**(b[ind] - 1)
            )
        )
