import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline as scipy_BSpline
from skfda.representation.basis import BSplineBasis

degree = 3
knots = np.linspace(0, 1, 5)
knots_with_boundary = np.r_[[knots[0]] * degree, knots, [knots[-1]] * degree]
x = np.linspace(0, 1, 50)

scipy_matrix = scipy_BSpline.design_matrix(
    x, knots_with_boundary, degree
).toarray()
# no need to add boundary knots (they are added automatically)
fda_basis = BSplineBasis(knots=knots, order=degree + 1)
fda_matrix = fda_basis.to_basis().to_grid(grid_points=x).data_matrix.squeeze()

# plot of basis functions
fig, ax = plt.subplots(1, 3, figsize=(12, 3))
ax[0].set_title("Basis functions")
for i, db in enumerate(scipy_matrix.T):
    ax[0].plot(db, color=f"C{i}", alpha=0.3, lw=2)
for i, d in enumerate(fda_matrix):
    ax[0].plot(d, color=f"C{i}", alpha=0.6, ls="--")
handles = [
    plt.Line2D([0], [0], color="gray", alpha=0.3, lw=2),
    plt.Line2D([0], [0], color="gray", alpha=0.6, lw=1, ls="--"),
]
labels = ["SCIPY", "FDA"]
ax[0].legend(handles, labels, loc="upper right")

ax[1].pcolor(scipy_matrix)
ax[1].set_title("SCIPY B-spline matrix")
ax[1].set_xlabel("Basis function")

ax[2].pcolor(fda_matrix.T)
ax[2].set_title("FDA B-spline matrix")
ax[2].set_xlabel("Basis function")
plt.tight_layout()
plt.savefig("bspline_basis_design_matrix.png")
