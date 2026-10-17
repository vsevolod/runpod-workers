# SPDX-License-Identifier: GPL-3.0-only
"""Flow-matching solvers for Qwen's Diffusers denoising loop.

ER-SDE is adapted from ComfyUI's sample_er_sde (GPL-3.0), commit
ce1e922d8623bb507180e32a0c01a7d0e29edef7, comfy/k_diffusion/sampling.py.
See ../LICENSES/COMFYUI-GPL-3.0.txt. Algorithm: arXiv:2309.06169.
Defaults match Comfy: max_stage=3, s_noise=1, 200 integration points.
"""

import math

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor

from .sampling import COMFY_SHIFT


class QwenHeunScheduler(FlowMatchHeunDiscreteScheduler):
    """Use upstream Heun steps with the same sigma transforms as Qwen's Euler."""

    def __init__(self, grid_scheduler):
        super().__init__(num_train_timesteps=grid_scheduler.config.num_train_timesteps)
        self.register_to_config(**dict(grid_scheduler.config))
        self._grid_scheduler = grid_scheduler

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None):
        grid = self._grid_scheduler
        grid.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
        self.num_inference_steps = grid.num_inference_steps
        # Heun evaluates each interior sigma twice: predictor, then corrector.
        self.timesteps = torch.cat([grid.timesteps[:1], grid.timesteps[1:].repeat_interleave(2)])
        self.sigmas = torch.cat([grid.sigmas[:1], grid.sigmas[1:-1].repeat_interleave(2), grid.sigmas[-1:]])
        self.prev_derivative = self.dt = self.sample = None
        self._step_index = self._begin_index = None


class QwenERSDEScheduler(FlowMatchEulerDiscreteScheduler):
    """Comfy ER-SDE-Solver-3 for rectified flow, consuming model velocity."""

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None):
        super().set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
        self.sigmas = self.sigmas.to(device=device).clone()
        if self.sigmas[0] >= 1:
            # CONST/Flux half-logSNR is singular at sigma=1. Comfy offsets the
            # first sigma by percent=1e-4 through ModelSamplingFlux(shift=.69).
            t = 1 - 1e-4
            shift = math.exp(COMFY_SHIFT)
            self.sigmas[0] = shift / (shift + (1 / t - 1))
            self.timesteps[0] = self.sigmas[0] * self.config.num_train_timesteps
        self._er_lambdas = self.sigmas.logit().exp()  # sigma / (1 - sigma)
        self._old_denoised = self._old_denoised_d = None
        self._integration_points = torch.arange(200, dtype=torch.float32, device=device)

    @staticmethod
    def _noise_scaler(value):
        return value * ((value ** .3).exp() + 10.)

    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        if self.step_index is None:
            self._init_step_index(timestep)
        index = self.step_index
        sigma, sigma_next = self.sigmas[index:index + 2]
        # The denoiser in Comfy's CONST model returns x - sigma * velocity.
        sample = sample.float()
        denoised = sample - sigma * model_output.float()
        if sigma_next == 0:
            prev_sample = denoised
        else:
            lambda_s, lambda_t = self._er_lambdas[index:index + 2]
            alpha_s, alpha_t = sigma / lambda_s, sigma_next / lambda_t
            scale_t = self._noise_scaler(lambda_t)
            ratio = scale_t / self._noise_scaler(lambda_s)
            prev_sample = (alpha_t / alpha_s) * ratio * sample + alpha_t * (1 - ratio) * denoised

            if index >= 1:
                dt = lambda_t - lambda_s
                step_size = -dt / 200.
                positions = lambda_t + self._integration_points * step_size
                scaled_positions = self._noise_scaler(positions)
                integral = (1 / scaled_positions).sum() * step_size
                derivative = (denoised - self._old_denoised) / (lambda_s - self._er_lambdas[index - 1])
                prev_sample = prev_sample + alpha_t * (dt + integral * scale_t) * derivative
                if index >= 2:
                    integral_u = ((positions - lambda_s) / scaled_positions).sum() * step_size
                    derivative_u = (derivative - self._old_denoised_d) / ((lambda_s - self._er_lambdas[index - 2]) / 2)
                    prev_sample = prev_sample + alpha_t * (dt.square() / 2 + integral_u * scale_t) * derivative_u
                self._old_denoised_d = derivative

            rng = generator if generator is not None else self.noise_generator
            if rng is None:
                raise ValueError("er_sde requires a request-scoped noise generator")
            noise = randn_tensor(sample.shape, generator=rng, device=sample.device, dtype=sample.dtype)
            noise_scale = (lambda_t.square() - lambda_s.square() * ratio.square()).sqrt().nan_to_num(nan=0.)
            prev_sample = prev_sample + alpha_t * noise * noise_scale

        self._old_denoised = denoised
        self._step_index += 1
        prev_sample = prev_sample.to(model_output.dtype)
        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
