# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
from typing import Union

import torch
import torch_harmonics as th

from earth2mip.time_loop import TimeLoop
import earth2mip.forecast_metrics_io
import numpy as np
import xarray as xr


class GaussianRandomFieldS2(torch.nn.Module):
    def __init__(
        self,
        nlat,
        alpha=2.0,
        tau=3.0,
        sigma=None,
        radius=1.0,
        grid="equiangular",
        dtype=torch.float32,
    ):
        super().__init__()
        """A mean-zero Gaussian Random Field on the sphere with Matern covariance:
        C = sigma^2 (-Lap + tau^2 I)^(-alpha).

        Lap is the Laplacian on the sphere, I the identity operator,
        and sigma, tau, alpha are scalar parameters.

        Note: C is trace-class on L^2 if and only if alpha > 1.

        Parameters
        ----------
        nlat : int
            Number of latitudinal modes;
            longitudinal modes are 2*nlat.
        alpha : float, default is 2
            Regularity parameter. Larger means smoother.
        tau : float, default is 3
            Lenght-scale parameter. Larger means more scales.
        sigma : float, default is None
            Scale parameter. Larger means bigger.
            If None, sigma = tau**(0.5*(2*alpha - 2.0)).
        radius : float, default is 1
            Radius of the sphere.
        grid : string, default is "equiangular"
            Grid type. Currently supports "equiangular" and
            "legendre-gauss".
        dtype : torch.dtype, default is torch.float32
            Numerical type for the calculations.
        """

        # Number of latitudinal modes.
        self.nlat = nlat

        # Default value of sigma if None is given.
        if sigma is None:
            assert alpha > 1.0, f"Alpha must be greater than one, got {alpha}."  # noqa
            sigma = tau ** (0.5 * (2 * alpha - 2.0))

        # Inverse SHT
        self.isht = th.InverseRealSHT(
            self.nlat, 2 * self.nlat, grid=grid, norm="backward"
        ).to(dtype=dtype)

        # Square root of the eigenvalues of C.
        sqrt_eig = (
            torch.tensor([j * (j + 1) for j in range(self.nlat)])
            .view(self.nlat, 1)
            .repeat(1, self.nlat + 1)
        )
        sqrt_eig = torch.tril(
            sigma * (((sqrt_eig / radius**2) + tau**2) ** (-alpha / 2.0))
        )
        sqrt_eig[0, 0] = 0.0
        sqrt_eig = sqrt_eig.unsqueeze(0)
        self.register_buffer("sqrt_eig", sqrt_eig)

        # Save mean and var of the standard Gaussian.
        # Need these to re-initialize distribution on a new device.
        mean = torch.tensor([0.0]).to(dtype=dtype)
        var = torch.tensor([1.0]).to(dtype=dtype)
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)

        # Standard normal noise sampler.
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

    def forward(self, N, xi=None):
        """Sample random functions from a spherical GRF.

        Parameters
        ----------
        N : int
            Number of functions to sample.
        xi : torch.Tensor, default is None
            Noise is a complex tensor of size (N, nlat, nlat+1).
            If None, new Gaussian noise is sampled.
            If xi is provided, N is ignored.

        Output
        -------
        u : torch.Tensor
           N random samples from the GRF returned as a
           tensor of size (N, nlat, 2*nlat) on a equiangular grid.
        """
        # Sample Gaussian noise.
        if xi is None:
            xi = self.gaussian_noise.sample(
                torch.Size((N, self.nlat, self.nlat + 1, 2))
            ).squeeze()
            xi = torch.view_as_complex(xi)

        # Karhunen-Loeve expansion.
        u = self.isht(xi * self.sqrt_eig)

        return u

    # Override cuda and to methods so sampler gets initialized with mean
    # and variance on the correct device.
    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)

        return self


def generate_noise_correlated(shape, *, reddening, device, noise_amplitude):
    return noise_amplitude * brown_noise(shape, reddening).to(device)


def generate_noise_grf(shape, grid, alpha, sigma, tau, device=None):
    sampler = GaussianRandomFieldS2(nlat=720, alpha=alpha, tau=tau, sigma=sigma).to(
        device
    )
    sample_noise = sampler(shape[0] * shape[1] * shape[2]).reshape(
        shape[0], shape[1], shape[2], 720, 1440
    )
    if grid.shape == (721, 1440):
        noise = torch.zeros(shape).to(device)
        noise[:, :, :, :-1, :] = sample_noise
        noise[:, :, :, -1:, :] = noise[:, :, :, -2:-1, :]
    else:
        noise = sample_noise
    return noise


def brown_noise(shape, reddening=2):
    noise = torch.normal(torch.zeros(shape), torch.ones(shape))

    x_white = torch.fft.fft2(noise)
    S = (
        torch.abs(torch.fft.fftfreq(noise.shape[-2]).reshape(-1, 1)) ** reddening
        + torch.abs(torch.fft.fftfreq(noise.shape[-1])) ** reddening
    )

    S = torch.where(S == 0, 0, 1 / S)
    S = S / torch.sqrt(torch.mean(S**2))

    x_shaped = x_white * S
    noise_shaped = torch.fft.ifft2(x_shaped).real

    return noise_shaped


def generate_bred_vector(
    x: torch.Tensor,
    model: TimeLoop,
    noise_amplitude: torch.Tensor,
    time: Union[datetime, None] = None,
    integration_steps: int = 30,
    inflate=False,
) -> torch.Tensor:
    # Assume x has shape [ENSEMBLE, TIME, CHANNEL, LAT, LON]

    if isinstance(noise_amplitude, float):
        noise_amplitude = torch.tensor([0.01]).to(x.device)

    assert (noise_amplitude.shape[0] == x.shape[2]) or (  # noqa
        torch.numel(noise_amplitude) == 1
    )
    coslat = torch.Tensor(np.cos(np.deg2rad(np.linspace(90,-90,721)))[None, None, :, None]).to(x.device)
    if np.asarray(model.grid.lat).shape[0] == 720:
        coslat = coslat[:, :, :-1]
    optimization_target = _load_optimal_targets('sfno_linear_73chq_sc3_layers8_edim384_wstgl2', 12).to(x.device).squeeze() / np.sqrt(2)  #/ model.scale.squeeze()
    optimization_target = optimization_target[None, None]
    x0 = x[:1]

    # Get control forecast
    num_steps = 0
    iterator = model(time, x0, yield_initial=False)
    #for _, data, _ in model(time, x0, yield_initial=False):
    for _, data, _ in iterator:
        xd = data
        num_steps += 1
        if num_steps == 1:
            break

    # Unsqueeze if time has been collapsed.
    if xd.ndim != x0.ndim:
        xd = xd.unsqueeze(1)

    dx = noise_amplitude[:, None, None] * torch.randn(
        x.shape, device=x.device, dtype=x.dtype
    )
    #dx = dx * model.scale
    dx = dx * torch.from_numpy(np.load("/pscratch/sd/j/jpathak/73var-6hourly/stats/time_means.npy").squeeze()).to(x.device)
    #optimization_target = (dx * coslat).std((-1,-2)) / 10
    for _ in range(integration_steps):
        x1 = x + dx
        num_steps = 0
        iterator = model(time, x1, yield_initial=False)
        #for _, data, _ in model(time, x1, yield_initial=False):
        for _, data, _ in iterator:
            x2 = data
            num_steps += 1
            if num_steps == 1:
                break

        # Unsqueeze if time has been collapsed.
        if x2.ndim != x1.ndim:
            x2 = x2.unsqueeze(1)
        dx = x2 - xd
        #dx = normalize(x2, model.center, model.scale) - normalize(xd, model.center, model.scale)

        if inflate:
            dx += noise_amplitude * (dx - dx.mean(dim=0))
    print(torch.linalg.norm(x, dim=(-1,-2)).shape)
    return 0.1 * dx * torch.linalg.norm(x, dim=(-1,-2))[:, :, None, None] / torch.linalg.norm(x+dx, dim=(-1,-2))[:, :, None, None] / model.scale
    exaggerate_factor = (dx * coslat).std((-1,-2)) / optimization_target
    dx = dx / exaggerate_factor[:, :, :, None, None]
    return dx / model.scale

def normalize(x, center, scale):
    return (x - center)/ scale


def TEMP_load_optimal_targets(config_model_name, lead_time):
    path = "/pscratch/sd/a/amahesh/hens/optimal_perturbation_targets/{}.nc".format(config_model_name)
    ds = earth2mip.forecast_metrics_io.read_metrics(path).to_xarray()
    #ds = xr.open_dataset(path)
    ds['lead_time'] = ds['lead_time'] / np.timedelta64(1, 'h')
    da = (ds.sel(metric='mse')**0.5).mean('initial_time')
    return torch.from_numpy(da.sel(lead_time=lead_time).values[np.newaxis,])

def _load_optimal_targets(config_model_name, lead_time):  
    path = "/pscratch/sd/a/amahesh/hens/optimal_perturbation_targets/{}.nc".format("FULL_sfno_73ch_e620_depth8_scale2_droppath01seed4_fcn-dev")
    ds = xr.open_dataset(path)                                          
    ds['lead_time'] = ds['lead_time'] / np.timedelta64(1, 'h')          
    return torch.from_numpy(ds['rmse'].sel(lead_time=lead_time).values[np.newaxis,])

