from copy import deepcopy, copy
import numpy as np


class BaseElement():
    """
    This is the abstract class for the creation of a BaseElement. A BaseElement
    does not have parameters or states.
    """

    # Executed when class is defined, not when an instance of the class is created, 
    # and are shared attributes of instances of this class
    _num_downstream = None
    """
    Number of downstream elements
    """

    _num_upstream = None
    """
    Number of upstream elements
    """

    input = {}
    """
    Dictionary of input fluxes
    """

    def __init__(self, id):
        """
        This is the initializer of the abstract class BaseElement.

        Parameters
        ----------
        id : str
            Identifier of the element. All the elements of the framework must
            have an identifier.
        """

        self.id = id
        self._error_message = 'module : superflexPy, Element : {},'.format(id)
        self._error_message += ' Error message : '

    @property
    def num_downstream(self):
        """
        Number of downstream elements.
        """

        return self._num_downstream

    @property
    def num_upstream(self):
        """
        Number of upstream elements
        """

        return self._num_upstream


class ParameterizedElement(BaseElement):
    """
    This is the abstract class for the creation of a ParameterizedElement. A
    ParameterizedElement has parameters but not states.
    """

    _prefix_parameters = ''
    """
    Prefix applied to the original names of the parameters
    """

    def __init__(self, parameters, id):
        """
        This is the initializer of the abstract class ParameterizedElement.

        Parameters
        ----------
        parameters : dict
            Parameters controlling the element. The parameters can be either
            a float (constant in time) or a numpy.ndarray of the same length
            of the input fluxes (time variant parameters).
        id : str
            Identifier of the element. All the elements of the framework must
            have an identifier.
        """

        BaseElement.__init__(self, id)

        self._parameters = parameters
        self.add_prefix_parameters(id)

    def get_parameters(self, names=None):
        """
        This method returns the parameters of the element.

        Parameters
        ----------
        names : list(str)
            Names of the parameters to return. The names must be the ones
            returned by the method get_parameters_name. If None, all the
            parameters are returned.

        Returns
        -------
        dict:
            Parameters of the element.
        """

        if names is None:
            return self._parameters
        else:
            return {n: self._parameters[n] for n in names}

    def get_parameters_name(self):
        """
        This method returns the names of the parameters of the element.

        Returns
        -------
        list(str):
            List with the names of the parameters.
        """

        return list(self._parameters.keys())

    def set_parameters(self, parameters):
        """
        This method sets the values of the parameters.

        Parameters
        ----------
        parameters : dict
            Contains the parameters of the element to be set. The keys must be
            the ones returned by the method get_parameters_name. Only the
            parameters that have to be changed should be passed.
        """

        for k in parameters.keys():
            if k not in self._parameters.keys():
                message = '{}The parameter {} does not exist'.format(self._error_message, k)
                raise KeyError(message)
            self._parameters[k] = parameters[k]

    def add_prefix_parameters(self, prefix):
        """
        This method add a prefix to the name of the parameters of the element.

        Parameters
        ----------
        prefix : str
            Prefix to be added. It cannot contain '_'.
        """

        if '_' in prefix:
            message = '{}The prefix cannot contain \'_\''.format(self._error_message)
            raise ValueError(message)

        # Extract the prefixes in the parameters name
        splitted = list(self._parameters.keys())[0].split('_')

        if prefix not in splitted:
            # Apply the prefix
            for k in list(self._parameters.keys()):
                value = self._parameters.pop(k)
                self._parameters['{}_{}'.format(prefix, k)] = value

            # Save the prefix for future uses
            self._prefix_parameters = '{}_{}'.format(prefix, self._prefix_parameters)


class StateElement(BaseElement):
    """
    This is the abstract class for the creation of a StateElement. A
    StateElement has states but not parameters.
    """

    _prefix_states = ''
    """
    Prefix applied to the original names of the parameters
    """

    def __init__(self, states, id):
        """
        This is the initializer of the abstract class StateElement.

        Parameters
        ----------
        states : dict
            Initial states of the element. Depending on the element the states
            can be either a float or a numpy.ndarray.
        id : str
            Identifier of the element. All the elements of the framework must
            have an id.
        """
        BaseElement.__init__(self, id)

        self._states = states
        self._init_states = deepcopy(states)  # It is used to re-set the states
        self.add_prefix_states(id)

    def get_states(self, names=None):
        """
        This method returns the states of the element.

        Parameters
        ----------
        names : list(str)
            Names of the states to return. The names must be the ones
            returned by the method get_states_name. If None, all the
            states are returned.

        Returns
        -------
        dict:
            States of the element.
        """

        if names is None:
            return self._states
        else:
            return {n: self._states[n] for n in names}

    def get_states_name(self):
        """
        This method returns the names of the states of the element.

        Returns
        -------
        list(str):
            List with the names of the states.
        """

        return list(self._states.keys())

    def set_states(self, states):
        """
        This method sets the values of the states.

        Parameters
        ----------
        states : dict
            Contains the states of the element to be set. The keys must be
            the ones returned by the method get_states_name. Only the
            states that have to be changed should be passed.
        """

        for k in states.keys():
            if k not in self._states.keys():
                message = '{}The state {} does not exist'.format(self._error_message, k)
                raise KeyError(message)
            self._states[k] = states[k]

    def add_prefix_states(self, prefix):
        """
        This method add a prefix to the id of the states of the element.

        Parameters
        ----------
        prefix : str
            Prefix to be added. It cannot contain '_'.
        """

        if '_' in prefix:
            message = '{}The prefix cannot contain \'_\''.format(self._error_message)
            raise ValueError(message)

        # Extract the prefixes in the parameters name
        splitted = list(self._states.keys())[0].split('_')

        if prefix not in splitted:
            # Apply the prefix
            for k in list(self._states.keys()):
                value = self._states.pop(k)
                self._states['{}_{}'.format(prefix, k)] = value

            # Save the prefix for furure uses
            self._prefix_states = '{}_{}'.format(prefix, self._prefix_states)


class StateParameterizedElement(StateElement, ParameterizedElement):
    """
    This is the abstract class for the creation of a StateParameterizedElement.
    A StateParameterizedElement has parameters and states.
    """

    def __init__(self, parameters, states, id):
        """
        This is the initializer of the abstract class
        StateParameterizedElement.

        Parameters
        ----------
        parameters : dict
            Parameters controlling the element. The parameters can be either
            a float (constant in time) or a numpy.ndarray of the same length
            of the input fluxes (time variant parameters).
        states : dict
            Initial states of the element. Depending on the element the states
            can be either a float or a numpy.ndarray.
        id : str
            Identifier of the element. All the elements of the framework must
            have an id.
        """

        StateElement.__init__(self, states, id)
        ParameterizedElement.__init__(self, parameters, id)


class ODEsElement(StateParameterizedElement):
    """
    This is the abstract class for the creation of a ODEsElement. An ODEsElement
    is an element with states and parameters that is controlled by an ordinary
    differential equation, of the form:

    dS/dt = input - output
    """

    _num_upstream = 1
    """
    Number of upstream elements
    """

    _num_downstream = 1
    """
    Number of downstream elements
    """

    _solver_states = []
    """
    List of states used by the solver of the differential equation
    """

    _fluxes = []
    """
    This attribute contains a list of methods (one per differential equation)
    that calculate the values of the fluxes needed to solve the differential
    equations that control the element. The single functions must return the
    fluxes as a list where incoming fluxes are positive and outgoing are
    negative. Here is a list of the required outputs of the single functions:

    list(floats)
        Values of the fluxes given states, inputs, and parameters.
    float
        Minimum value of the state. Used, sometimes, by the numerical solver
        to search for the solution.
    float
        Maximum value of the state. Used, sometimes, by the numerical solver
        to search for the solution.
    list(floats)
        Values of the derivatives of the fluxes w.r.t. the states.
    """

    def __init__(self, parameters, states, approximation, id):
        """
        This is the initializer of the abstract class ODEsElement.

        Parameters
        ----------
        parameters : dict
            Parameters controlling the element. The parameters can be either
            a float (constant in time) or a numpy.ndarray of the same length
            of the input fluxes (time variant parameters).
        states : dict
            Initial states of the element. Depending on the element the states
            can be either a float or a numpy.ndarray.
        approximation : superflexpy.utils.numerical_approximation.NumericalApproximator
            Numerial method used to approximate the differential equation
        id : str
            Identifier of the element. All the elements of the framework must
            have an id.
        """

        StateParameterizedElement.__init__(self, parameters=parameters,
                                           states=states, id=id)

        self._num_app = approximation

    def set_timestep(self, dt):
        """
        This method sets the timestep used by the element.

        Parameters
        ----------
        dt : float
            Timestep
        """
        self._dt = dt

    def define_numerical_approximation(self, approximation):
        """
        This method define the solver to use for the differential equation.

        Parameters
        ----------
        solver : superflexpy.utils.root_finder.RootFinder
            Solver used to find the root(s) of the differential equation(s).
            Child classes may implement their own solver, therefore the type
            of the solver is not enforced.
        """

        self._num_app = approximation

    def _solve_differential_equation(self, **kwargs):
        """
        This method calls the solver of the differential equation(s). When
        called, it solves the differential equation(s) for all the timesteps
        and populates self.state_array.
        """

        if len(self._solver_states) == 0:
            message = '{}the attribute _solver_states must be filled'.format(self._error_message)
            raise ValueError(message)

        self.state_array = self._num_app.solve(fun=self._fluxes,
                                               S0=self._solver_states,
                                               dt=self._dt,
                                               **self.input,
                                               **{k[len(self._prefix_parameters):]: self._parameters[k] for k in self._parameters},
                                               **kwargs)
