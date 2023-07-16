import numpy as np
from skfda.representation.basis import BSplineBasis
import matplotlib.pyplot as plt
import skfda

from scipy.interpolate import make_lsq_spline, BSpline


def get_bspline_matrix(knots: np.ndarray, degree=3):
    """Generate a B-spline density basis of any degree"""
    knots_with_boundary = np.r_[[knots[0]] * degree, knots, [knots[-1]] * degree]
    n_knots = len(knots_with_boundary)  # number of knots (including the external knots)
    assert n_knots == degree * 2 + len(knots)
    x = np.linspace(0, 1, 50) # points to evaluate the basis functions at
    return BSpline.design_matrix(x, knots_with_boundary, degree)





knots = np.linspace(0, 1, 30)
degree = 3
bspline_matrix = get_bspline_matrix(knots, degree)
plt.imshow(bspline_matrix.toarray())
plt.show()


spline_object = make_lsq_spline(knots, bspline_matrix.toarray(), t=np.linspace(0, 1, 50), k=degree)

#
#
# k = 30
# degree = 3
# diffMatrixOrder = 1
# knots = np.linspace(0, 1, k - degree + 1)
# fda_basis = BSplineBasis(knots=knots, order=degree)
# fda_basis.plot()
# plt.suptitle('Basis functions 1')
# plt.show()
#
#
# data = fda_basis.to_basis().to_grid().data_matrix
# for d in data:
#     plt.plot(d)
# plt.suptitle('Basis functions 3')
# plt.show()
#
#
#
# new_basis = BSplineBasis()
# new_basis.plot()
