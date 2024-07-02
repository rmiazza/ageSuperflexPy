class LumpedModel():
    """
    This class defines a lumped hydrologycal model (i.e. Unit). A unit can is a
    collection of elements. It's task is to build the basic structure,
    connecting different elements. Mathematically, it is a directed acyclic
    graph.
    """

    def __init__(self, layers):
        """
        This is the initializer of the class LumpedModel.

        Parameters
        ----------
        layers : list(list(superflexpy.framework.element.BaseElement))
            This list defines the structure of the model. The elements are
            arranged in layers (upstream to downstream) and each layer can
            contain multiple elements.
        """
        self._error_message = 'module : superflexPy, LumpedModel,'
        self._error_message += ' Error message : '

        self._layers = layers

        self._check_layers()
        self._construct_dictionary()

    # METHODS FOR THE USER

    def set_input(self, input):
        """
        This method sets the inputs to the unit.

        Parameters
        ----------
        input : list(numpy.ndarray)
            List of input fluxes.
        """

        self.input = input

    def get_output(self, solve=True):
        """
        This method solves the Unit, solving each Element and putting together
        their outputs according to the structure.

        Parameters
        ----------
        solve : bool
            True if the elements have to be solved (i.e. calculate the states).

        Returns
        -------
        list(numpy.ndarray)
            List containing the output fluxes of the unit.
        """

        # Set the first layer (it must have 1 element)
        self._layers[0][0].set_input(self.input)

        for i in range(1, len(self._layers)):
            # Collect the outputs
            outputs = []
            for el in self._layers[i - 1]:
                if el.num_downstream == 1:
                    outputs.append(el.get_output(solve))
                else:
                    loc_out = el.get_output(solve)
                    for o in loc_out:
                        outputs.append(o)

            # Fill the inputs
            ind = 0
            for el in self._layers[i]:
                if el.num_upstream == 1:
                    el.set_input(outputs[ind])
                    ind += 1
                else:
                    loc_in = []
                    for _ in range(el.num_upstream):
                        loc_in.append(outputs[ind])
                        ind += 1
                    el.set_input(loc_in)

        # Return the output of the last element
        return self._layers[-1][0].get_output(solve)

    def get_internal(self, id, attribute):
        """
        This method allows to inspect attributes of the objects that belong to
        the unit.

        Parameters
        ----------
        id : str
            Id of the object.
        attribute : str
            Name of the attribute to expose.

        Returns
        -------
        Unknown
            Attribute exposed
        """

        return self._find_attribute_from_name(id, attribute)

    def call_internal(self, id, method, **kwargs):
        """
        This method allows to call methods of the objects that belong to the
        unit.

        Parameters
        ----------
        id : str
            Id of the object.
        method : str
            Name of the method to call.

        Returns
        -------
        Unknown
            Output of the called method.
        """

        method = self._find_attribute_from_name(id, method)
        return method(**kwargs)

    # PROTECTED METHODS

    def _construct_dictionary(self):
        """
        This method populates the self._content_pointer dictionary.
        """

        self._content_pointer = {}

        for i in range(len(self._layers)):
            for j in range(len(self._layers[i])):
                if self._layers[i][j].id in self._content_pointer:
                    message = '{}The element {} already exist.'.format(self._error_message, self._layers[i][j].id)
                    raise KeyError(message)
                self._content_pointer[self._layers[i][j].id] = (i, j)

        self._content = {}
        for k in self._content_pointer.keys():
            l, el = self._content_pointer[k]
            self._content[(l, el)] = self._layers[l][el]

    def _find_attribute_from_name(self, id, function):
        """
        This method is used to find the attributes or methods of the components
        contained for post-run inspection.

        Parameters
        ----------
        id : str
            Identifier of the component
        function : str
            Name of the attribute or method

        Returns
        -------
        Unknown
            Attribute or method to inspect
        """

        # Search the element
        (l, el) = self._find_content_from_name(id)
        element = self._layers[l][el]

        # Call the function on the element
        try:
            method = getattr(element, function)
        except AttributeError:
            message = '{}the method {} does not exist.'.format(self._error_message, function)
            raise AttributeError(message)

        return method

    def _find_content_from_name(self, name):
        """
        This method finds a component using the name of the parameter or the
        state.

        Parameters
        ----------
        name : str
            Name to use for the search

        Returns
        -------
        int or tuple
            Index of the component in self._content
        """

        splitted_name = name.split('_')

        try:
            class_id = self.id
        except AttributeError:  # We are in a Model
            class_id = None

        if class_id is not None:
            if class_id in splitted_name:
                if (len(splitted_name) - splitted_name.index(class_id)) == 2:  # It is a local parameter
                    position = -1
                else:
                    # HRU or Catchment
                    ind = splitted_name.index(class_id)
                    position = self._content_pointer[splitted_name[ind + 1]]
            else:
                position = self._content_pointer[splitted_name[0]]
        else:
            # Network. The network doesn't have local parameters
            for c in self._content_pointer.keys():
                if c in splitted_name:
                    position = self._content_pointer[c]
                    break
                else:
                    position = None

        return position

    def _check_layers(self):
        """
        This method controls if the layers respect all the rules in terms of
        number of upstream/downstream elements.
        """

        # Check layer 0
        if len(self._layers[0]) != 1:
            message = '{}layer 0 has {} elements.'.format(self._error_message, len(self._layers[0]))
            raise ValueError(message)

        if self._layers[0][0].num_upstream != 1:
            message = '{}The element in layer 0 has {} upstream elements.'.format(self._error_message, len(self._layers[0][0].num_upstream))
            raise ValueError(message)

        # Check the other layers
        for i in range(1, len(self._layers)):
            num_upstream = 0
            num_downstream = 0
            for el in self._layers[i - 1]:
                num_downstream += el.num_downstream
            for el in self._layers[i]:
                num_upstream += el.num_upstream

            if num_downstream != num_upstream:
                message = '{}Downstream : {}, Upstream : {}'.format(self._error_message, num_downstream, num_upstream)
                raise ValueError(message)

        # Check last layer
        if len(self._layers[-1]) != 1:
            message = '{}last layer has {} elements.'.format(self._error_message, len(self._layers[-1]))
            raise ValueError(message)

        if self._layers[-1][0].num_downstream != 1:
            message = '{}The element in the last layer has {} downstream elements.'.format(self._error_message, len(self._layers[-1][0].num_downstream))
            raise ValueError(message)

    def set_timestep(self, dt):
        """
        This method sets the timestep used by the element.

        Parameters
        ----------
        dt : float
            Timestep
        """

        self._dt = dt

        for c in self._content_pointer.keys():
            position = self._content_pointer[c]

            try:
                self._content[position].set_timestep(dt)
            except AttributeError:
                continue
