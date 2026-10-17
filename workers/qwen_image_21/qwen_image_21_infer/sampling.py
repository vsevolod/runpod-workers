"""Sampling policy and request-scoped scheduler construction."""

import math

# ComfyUI supported_models.QwenImage21 uses ModelSamplingFlux(shift=.69).
# ModelSamplingFlux builds 10000 sigmas; simple_scheduler samples that table.
COMFY_SHIFT = .69
COMFY_TIMESTEPS = 10000


def comfy_simple_sigmas(steps: int = 40) -> list[float]:
    shift = math.exp(COMFY_SHIFT)
    sigmas = []
    for index in range(steps):
        timestep = (COMFY_TIMESTEPS - int(index * COMFY_TIMESTEPS / steps)) / COMFY_TIMESTEPS
        sigmas.append(shift / (shift + (1 / timestep - 1)))
    # Diffusers appends terminal zero itself; return exactly `steps` values.
    return sigmas


def base_scheduler(scheduler):
    from diffusers import FlowMatchEulerDiscreteScheduler

    # The supplied grid already includes Comfy's shift. Disable the upstream
    # resolution-dependent shift and terminal stretch to avoid applying them twice.
    return FlowMatchEulerDiscreteScheduler.from_config(
        scheduler.config, use_dynamic_shifting=False, shift=1., shift_terminal=None,
        use_karras_sigmas=False, use_exponential_sigmas=False, use_beta_sigmas=False,
        invert_sigmas=False, stochastic_sampling=False,
    )


def make_scheduler(template, sampler: str, schedule: str, *, generator=None):
    """A fresh solver for each request; preserve the checkpoint's grid transforms."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    grid = base_scheduler(template) if schedule == "simple" else FlowMatchEulerDiscreteScheduler.from_config(template.config)
    if sampler == "euler":
        return grid
    from .schedulers import QwenHeunScheduler, QwenERSDEScheduler

    if sampler == "heun":
        return QwenHeunScheduler(grid)
    if sampler == "er_sde":
        scheduler = QwenERSDEScheduler.from_config(grid.config)
        scheduler.noise_generator = generator
        return scheduler
    raise ValueError(f"Unsupported sampler: {sampler!r}")
