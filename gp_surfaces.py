from typing import *
from dataclasses import dataclass

import scipy.stats as ss
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
import gpytorch as gpyt
import gpytorch.utils.memoize


@dataclass
class PointsOnSurface:
    inputs: torch.FloatTensor  # [num_grid_points x surface_dim]
    targets: torch.FloatTensor  # [space_dim x num_grid_points]

    def __post_init__(self):
        assert self.inputs.shape[0] == self.targets.shape[1]


class GridGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, inputs, targets, kernel: gpyt.kernels.Kernel,
                 input_dims: int, grid_size: Union[int, List[int]],
                 grid_bounds=None):
        super(GridGPRegressionModel, self).__init__(inputs, targets, gpytorch.likelihoods.GaussianLikelihood())
        if grid_bounds is not None:
            grid_bounds_ = tuple(grid_bounds for _ in range(input_dims))
        else:
            grid_bounds_ = None
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(kernel, grid_size=grid_size, num_dims=input_dims,
                                                                     grid_bounds=grid_bounds_)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPSurface:

    def __init__(self, kernel: gpyt.kernels.Kernel,
                 space_dim: int, surface_dim: int, grid_size: Union[int, List[int]],
                 surface: Optional[PointsOnSurface] = None, grid_bounds: Tuple[float, float] = (0., 1.)):
        assert (0 < surface_dim < space_dim)
        self.kernel = kernel
        self.space_dim = space_dim
        self.surface_dim = surface_dim
        self.grid_size = grid_size
        self.grid_bounds = grid_bounds
        if surface is None:
            self._surface = GPSurface.sample_surface(self.kernel, self.space_dim,
                                                     self.surface_dim, self.grid_size,
                                                     self.grid_bounds)
        else:
            self._surface = surface
        self._gp_models = self.fit_surface(self._surface)

        # Make a call so that the covariance matrix is evaluated an cached.
        # This will make later surface evaluations very fast
        dummy_input = 0.5 * torch.ones((1, self.surface_dim))
        self.surface_center = self.evaluate_surface(dummy_input)

    @staticmethod
    def sample_surface(kernel, space_dim, surface_dim, grid_size, grid_bounds) -> PointsOnSurface:
        # Get training data to None => prior mode
        tmp_model = GridGPRegressionModel(inputs=None, targets=None, kernel=kernel,
                                          input_dims=surface_dim, grid_size=grid_size, grid_bounds=grid_bounds).eval()
        surface_inputs = make_meshgrid(grid_size, surface_dim, grid_bounds)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mvn_prior = tmp_model(surface_inputs)

        surface_targets = mvn_prior.sample(torch.Size((space_dim,)))

        return PointsOnSurface(surface_inputs, surface_targets)

    def fit_surface(self, surface: PointsOnSurface):
        assert (surface.inputs.shape[-1] == self.surface_dim)
        assert (surface.targets.shape[0] == self.space_dim)
        gp_models = [GridGPRegressionModel(inputs=surface.inputs, targets=surface.targets[i],
                                           kernel=self.kernel, input_dims=self.surface_dim,
                                           grid_size=self.grid_size) for i in range(self.space_dim)]
        [model_.eval() for model_ in gp_models]

        return gp_models

    def evaluate_surface(self, inputs):
        n, dim = inputs.shape
        assert (dim == self.surface_dim)
        preds = torch.zeros((self.space_dim, n))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i, model in enumerate(self._gp_models):
                preds[i] = model(inputs).loc
        return preds

    def __call__(self, inputs):
        return self.evaluate_surface(inputs)

    def plot_surface(self, **kwargs):
        if self.surface_dim == 2 and self.space_dim == 3:
            return plot_surface(self._surface.targets.view(self.space_dim, self.grid_size, self.grid_size).numpy(),
                                **kwargs)
        elif self.surface_dim == 1 and self.space_dim < 4:
            return plot_surface(self._surface.targets.view(self.space_dim, self.grid_size).numpy(), **kwargs)


def sample_uniform_from_subspace(num_samples, space_dim, num_subspace_dim,
                                 rot_matrix=None, subspace_dims=None):
    assert (0 < num_subspace_dim < space_dim)
    samples = torch.rand((num_samples, num_subspace_dim)) - 0.5
    samples = _apply_subspace_transform(samples, space_dim, rot_matrix, subspace_dims)
    return samples


def _apply_subspace_transform(subspace_values, space_dim, rot_matrix=None, subspace_dims=None):
    assert (subspace_values.abs().max() < 0.5 + 1e-3)
    num_subspace_dim = subspace_values.shape[-1]
    subspace_values = _grid_padding(subspace_values, space_dim - num_subspace_dim)
    if rot_matrix is not None:
        subspace_values = torch.matmul(subspace_values, rot_matrix)
    if subspace_dims is not None:
        index = _mk_permutation_index(space_dim, subspace_dims)
        subspace_values = torch.index_select(subspace_values, dim=1, index=torch.LongTensor(index))
    return subspace_values + 0.5


def sample_random_rotation_matrix(space_dim):
    return torch.from_numpy(ss.special_ortho_group.rvs(space_dim)).float()


def _grid_padding(grid, padding):
    return torch.cat([grid, torch.zeros((grid.shape[0], padding))], dim=1)


def _mk_permutation_index(num_dims, permutation_dims):
    num_permuation_dims = len(permutation_dims)
    index = list(range(0, num_dims))
    try:
        for i, sub_dim in zip(range(num_permuation_dims), permutation_dims):
            index[sub_dim] = i
            index[i] = sub_dim
    except TypeError:
        print(f'Invalid input {permutation_dims} for subspace dims.')
        raise
    return index


def make_meshgrid(grid_size: int, space_dim: int, grid_bounds: Tuple[float, float]) -> torch.FloatTensor:
    return torch.stack([vec.reshape(-1) for vec in
                        torch.meshgrid(
                            *torch.linspace(grid_bounds[0], grid_bounds[1], grid_size).expand(space_dim, grid_size))],
                       dim=1).to(torch.float)


def get_kernel(kernel_name, lengthscale=1):
    if kernel_name.lower() == 'rbf':
        kernel = gpytorch.kernels.RBFKernel().initialize(lengthscale=lengthscale)
    elif kernel_name.lower() == 'periodic':
        return gpytorch.kernels.PeriodicKernel().initialize(lengthscale=lengthscale, period_length=1)
    elif kernel_name.lower() == 'cosine':
        kernel = gpytorch.kernels.CosineKernel().initialize(period_length=1)
    else:
        raise ValueError(f"Unknown kernel {kernel_name}")

    # Due to some 'bug' in gpytorch, ScaleKernel have to be used with GridInterpolationKernel
    return gpytorch.kernels.ScaleKernel(kernel)


def plot_surface(data, **kwargs):
    dim = data.shape[0]
    grid_dims = data.shape[1:]
    assert (1 <= len(grid_dims) < dim < 4)
    fig = plt.figure()
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        if len(grid_dims) == 2:
            ax.plot_wireframe(*data, **kwargs)
        elif len(grid_dims) == 1:
            ax.plot(*data, **kwargs)
    else:
        ax = fig.add_subplot(111)
        if len(grid_dims) == 2:
            ax.plot_wireframe(*data, **kwargs)
        elif len(grid_dims) == 1:
            ax.plot(*data, **kwargs)
    return fig, ax
