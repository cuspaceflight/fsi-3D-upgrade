"""
This module provides a mechanism to imterpolate point data acquired from preCICE into FEniCS Expressions.
"""

from dolfin import UserExpression
from .adapter_core import FunctionType
from scipy.interpolate import Rbf
from scipy.linalg import lstsq
import numpy as np
from mpi4py import MPI

import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class CouplingExpression(UserExpression):
    """
    Creates functional representation (for FEniCS) of nodal data provided by preCICE.
    """

    def set_function_type(self, function_type):
        self._function_type = function_type

    def update_boundary_data(self, vals, coords_x, coords_y=None, coords_z=None):
        """
        Update object of this class of type FEniCS UserExpression with given point data.

        Parameters
        ----------
        vals : double
            Point data to be used to update the Expression.
        coords_x : double
            X coordinate of points of which point data is provided.
        coords_y : double
            Y coordinate of points of which point data is provided.
        coords_z : double
            Z coordinate of points of which point data is provided.
        """
        self._coords_x = coords_x
        self._dimension = 3
        if coords_y is None:
            self._dimension -= 1
            coords_y = np.zeros(self._coords_x.shape)
        if coords_z is None:
            self._dimension -= 1
            coords_z = np.zeros(self._coords_x.shape)

        self._coords_y = coords_y
        self._coords_z = coords_z
        self._vals = vals

        self._f = self.create_interpolant()

        if self.is_scalar_valued():
            assert (self._vals.shape == self._coords_x.shape)
        elif self.is_vector_valued():
            assert (self._vals.shape[0] == self._coords_x.shape[0])

    def interpolate(self, x):
        # TODO: the correct way to deal with this would be using an abstract class. Since this is technically more
        # complex and the current implementation is a workaround anyway, we do not
        # use the proper solution, but this hack.
        """
        Interpolates at x. Uses buffered interpolant self._f.
        Parameters
        ----------
        x : double
            Point.

        Returns
        -------
        list : python list
            A list containing the interpolated values. If scalar function is interpolated this list has a single
            element. If a vector function is interpolated the list has self._dimensions elements.
        """
        raise Exception("Please use one of the classes derived from this class, that implements an actual strategy for"
                        "interpolation.")

    def create_interpolant(self, x):
        # TODO: the correct way to deal with this would be using an abstract class. Since this is technically more
        # complex and the current implementation is a workaround anyway, we do not
        # use the proper solution, but this hack.
        """
        Creates interpolant from boundary data that has been provided before.

        Parameters
        ----------
        x : double
            Point.

        Returns
        -------
        list : python list
            Interpolant as a list. If scalar function is interpolated this list has a single
            element. If a vector function is interpolated the list has self._dimensions elements.
        """
        raise Exception("Please use one of the classes derived from this class, that implements an actual strategy for"
                        "interpolation.")

    def eval(self, value, x):
        """
        Evaluates expression at x using self.interpolate(x) and stores result to value.

        Parameters
        ----------
        value : double
            Buffer where result has to be returned to.
        x : double
            Coordinate where expression has to be evaluated.
        """
        return_value = self.interpolate(x)
        for i in range(self._vals.ndim):
            value[i] = return_value[i]

    def is_scalar_valued(self):
        """
        Determines if function being interpolated is scalar-valued based on dimension of provided vector self._vals.

        Returns
        -------
        tag : bool
            True if function being interpolated is scalar-valued, False otherwise.
        """
        try:
            if self._vals.ndim == 1:
                assert(self._function_type is FunctionType.SCALAR)
                return True
            elif self._vals.ndim > 1:
                assert(self._function_type is FunctionType.VECTOR)
                return False
            else:
                raise Exception("Dimension of the function is 0 or negative!")
        except AttributeError:
            return self._function_type is FunctionType.SCALAR

    def is_vector_valued(self):
        """
        Determines if function being interpolated is vector-valued based on dimension of provided vector self._vals.

        Returns
        -------
        tag : bool
            True if function being interpolated is vector-valued, False otherwise.
        """
        try:
            if self._vals.ndim > 1:
                assert(self._function_type is FunctionType.VECTOR)
                return True
            elif self._vals.ndim == 1:
                assert(self._function_type is FunctionType.SCALAR)
                return False
            else:
                raise Exception("Dimension of the function is 0 or negative!")
        except AttributeError:
            return self._function_type is FunctionType.VECTOR


class SegregatedRBFInterpolationExpression(CouplingExpression):
    """
    Uses polynomial quadratic fit + RBF interpolation for implementation of CustomExpression.interpolate. Allows for
    arbitrary coupling interfaces.

    See Lindner, F., Mehl, M., & Uekermann, B. (2017). Radial basis function interpolation for black-box multi-physics
    simulations.
    """

    def segregated_interpolant_3d(self, coords_x, coords_y, coords_z, data):
        assert(coords_x.shape == coords_y.shape)
        # create least squares system to approximate a * x ** 2 + b * x + c ~= y

        # for 3D: 2nd order approximation of f(x,y,z)
        def lstsq_interp(x, y, z, w): 
            return w[0] * x ** 2 + w[1] * y ** 2 + w[2] * z ** 2 + w[3] * x * y + w[4] * x * z + w[5] * y * z + \
            w[6] * x + w[7] * y + w[8] * z + w[9]
        n_unknowns = 10
        # 1st order approximation
        # def lstsq_interp(x, y, z, w): 
        #     return w[0] * x + w[1] * y + w[2] * z + w[3]
        # n_unknowns = 4

        A = np.empty((coords_x.shape[0], 0))
        for i in range(n_unknowns):
            w = np.zeros([n_unknowns])
            w[i] = 1
            column = lstsq_interp(coords_x, coords_y, coords_z, w).reshape((coords_x.shape[0], 1))
            A = np.hstack([A, column])

        # solve system
        w, _, _, _ = lstsq(A, data)
        # create fit

        # compute remaining error
        res = data - lstsq_interp(coords_x, coords_y, coords_z, w)
        # add RBF for error
        rbf_interp = Rbf(coords_x, coords_y, coords_z, res)

        return lambda x, y, z: rbf_interp(x, y, z) + lstsq_interp(x, y, z, w) 

    def segregated_interpolant_2d(self, coords_x, coords_y, data):
        assert(coords_x.shape == coords_y.shape)
        # create least squares system to approximate a * x ** 2 + b * x + c ~= y

        # for 2D: 2nd order approximation of f(x,y)
        def lstsq_interp(x, y, w): return w[0] * x ** 2 + w[1] * y ** 2 + w[2] * x * y + w[3] * x + w[4] * y + w[5]

        A = np.empty((coords_x.shape[0], 0))
        n_unknowns = 6
        for i in range(n_unknowns):
            w = np.zeros([n_unknowns])
            w[i] = 1
            column = lstsq_interp(coords_x, coords_y, w).reshape((coords_x.shape[0], 1))
            A = np.hstack([A, column])

        # solve system
        w, _, _, _ = lstsq(A, data)
        # create fit

        # compute remaining error
        res = data - lstsq_interp(coords_x, coords_y, w)
        # add RBF for error
        rbf_interp = Rbf(coords_x, coords_y, res)

        return lambda x, y: rbf_interp(x, y) + lstsq_interp(x, y, w)

    def create_interpolant(self):
        """
        See base class description.
        """
        assert (self._dimension == 2 or self._dimension == 3) # only support 2D or 3D
        interpolant = []

        if self._dimension == 2:
            if self.is_scalar_valued():  # check if scalar or vector-valued
                interpolant.append(self.segregated_interpolant_2d(self._coords_x, self._coords_y, self._vals))
            elif self.is_vector_valued():
                for d in range(2):
                    interpolant.append(self.segregated_interpolant_2d(self._coords_x, self._coords_y, self._vals[:, d]))
                    
            else:
                raise Exception("Problem dimension and data dimension not matching.")
        else:
            if self.is_scalar_valued():  # check if scalar or vector-valued
                interpolant.append(self.segregated_interpolant_3d(self._coords_x, self._coords_y, self._coords_z, self._vals))
            elif self.is_vector_valued():
                for d in range(3):
                    interpolant.append(self.segregated_interpolant_3d(self._coords_x, self._coords_y, self._coords_z, self._vals[:, d]))
                    
            else:
                raise Exception("Problem dimension and data dimension not matching.")


        return interpolant

    def interpolate(self, x):
        """
        See base class description.
        """
        assert (self._dimension == 2 or self._dimension == 3) # only support 2D or 3D

        return_value = self._vals.ndim * [None]

        if self._dimension == 2:
            for i in range(self._vals.ndim):
                return_value[i] = self._f[i](x[0], x[1])
        else:
            for i in range(self._vals.ndim):
                return_value[i] = self._f[i](x[0], x[1], x[2])

        return return_value


class EmptyExpression(CouplingExpression):
    """A dummy expression that can be used for implementing a coupling boundary condition, if the participant's mesh has
    no vertices on the coupling domain. Only used for parallel runs.

    Example:
    We want solve
    F = u * v / dt * dx + dot(grad(u), grad(v)) * dx - (u_n / dt + f) * v * dx + v * coupling_expression * ds
    The user defines F, but does not know whether the rank even has vertices on the Neumann coupling boundary.
    If the rank does not have any vertices on the Neumann coupling boundary the coupling_expression is an
    EmptyExpression. This "deactivates" the Neumann BC for that specific rank.
    """

    def eval(self, value, x):
        """ Evaluates expression at x. For EmptyExpression always returns zero.

        :param x: coordinate where expression has to be evaluated
        :param value: buffer where result has to be returned to
        """
        assert(MPI.COMM_WORLD.Get_size() > 1)
        for i in range(self._vals.ndim):
            value[i] = 0

    def update_boundary_data(self, vals, coords_x, coords_y=None, coords_z=None):
        pass  # an EmptyExpression is never updated
