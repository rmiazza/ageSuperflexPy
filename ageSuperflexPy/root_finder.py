class RootFinder():
    """
    This is the abstract class for the creation of a RootFinder. It defines how
    the solver of the differential equation must be implemented.
    """

    architecture = None
    """
    Implementation required to increase the performance (e.g. numba)
    """

    def __init__(self, tol_F=1e-8, tol_x=1e-8, iter_max=10):
        """
        The constructor of the subclass must accept the parameters of the
        solver.

        Parameters
        ----------
        tol_F : float
            Tolerance on the y axis (distance from 0) that stops the solver
        tol_x : float
            Tolerance on the x axis (distance between two roots) that stops
            the solver
        iter_max : int
            Maximum number of iteration of the solver. After this value it
            raises a runtime error
        """

        self._tol_F = tol_F
        self._tol_x = tol_x
        self._iter_max = iter_max
        self._name = 'Solver'

    def get_settings(self):
        """
        This method returns the settings of the root finder.

        Returns
        -------
        float
            Function tollerance (tol_F)
        float
            X tollerance (tol_x)
        int
            Maximum number of iterations (iter_max)
        """

        return (
            self._tol_F,
            self._tol_x,
            self._iter_max,
        )
