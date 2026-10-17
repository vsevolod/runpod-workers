"""Numerical parity with ComfyUI's QwenImage21 Flux/simple schedule."""

import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1]))


class SamplingTests(unittest.TestCase):
    def sampling(self):
        try:
            from qwen_image_21_infer import sampling
        except ImportError:
            self.fail("Base Euler/simple sampling is not implemented")
        return sampling

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers"), "Install worker dependencies")
    def test_heun_uses_same_sigma_grid_and_improves_second_order_accuracy(self):
        import math
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        sampling = self.sampling()
        errors = {}
        for sampler in ("euler", "heun"):
            scheduler = sampling.make_scheduler(FlowMatchEulerDiscreteScheduler(), sampler, "simple")
            scheduler.set_timesteps(sigmas=sampling.comfy_simple_sigmas(25), mu=.7)
            self.assertEqual(len(scheduler.timesteps), 25 if sampler == "euler" else 49)
            self.assertAlmostEqual(float(scheduler.sigmas[0]), 1.)
            sample = torch.ones(1, 4)
            for t in scheduler.timesteps:
                sample = scheduler.step(sample, t, sample, return_dict=False)[0]
            self.assertTrue(torch.isfinite(sample).all())
            errors[sampler] = abs(float(sample[0, 0]) - math.exp(-1))
        # The final interval ends at zero with Euler, as in Comfy/Heun.
        self.assertLess(errors["heun"], errors["euler"] / 5)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers"), "Install worker dependencies")
    def test_er_sde_matches_pinned_comfy_reference_at_every_step(self):
        import math
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        from fixtures import comfy_er_sde as reference
        sampling = self.sampling()
        for steps in (1, 2, 8, 40, 200):
            with self.subTest(steps=steps):
                grid = sampling.comfy_simple_sigmas(steps)
                states = []
                def velocity(x, sigma):
                    return x * .15 + sigma.square() - .02
                class FlowSampling(reference.CONST):
                    def percent_to_sigma(self, percent):
                        t = 1 - percent
                        shift = math.exp(.69)
                        return shift / (shift + (1 / t - 1))
                class Model:
                    inner_model = SimpleNamespace(model_patcher=SimpleNamespace(get_model_object=lambda _: FlowSampling()))
                    def __call__(self, x, sigma, **kwargs):
                        return x - sigma * velocity(x, sigma)
                initial = torch.linspace(-1, 1, 12).reshape(1, 3, 4)
                ref_rng = torch.Generator().manual_seed(1234)
                expected = reference.sample_er_sde(
                    Model(), initial.clone(), torch.tensor(grid + [0.]),
                    noise_sampler=lambda *args: torch.randn(initial.shape, generator=ref_rng),
                    callback=lambda args: states.append(args["x"].clone()),
                )
                scheduler = sampling.make_scheduler(
                    FlowMatchEulerDiscreteScheduler(), "er_sde", "simple",
                    generator=torch.Generator().manual_seed(1234),
                )
                scheduler.set_timesteps(sigmas=grid, mu=.7)
                sample = initial.clone()
                for index, t in enumerate(scheduler.timesteps):
                    torch.testing.assert_close(sample, states[index], atol=3e-5, rtol=3e-5)
                    sigma = t / 1000
                    sample = scheduler.step(velocity(sample, sigma), t, sample, return_dict=False)[0]
                torch.testing.assert_close(sample, expected, atol=3e-5, rtol=3e-5)
                self.assertTrue(torch.isfinite(sample).all())

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers"), "Install worker dependencies")
    def test_custom_samplers_preserve_checkpoint_shifting_and_reinitialize_state(self):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        sampling = self.sampling()
        grid = [1., .978453, .954180, .926626, .895080, .845148, .704534, .414568]
        template = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, shift_terminal=.02)
        euler = sampling.make_scheduler(template, "euler", "checkpoint")
        euler.set_timesteps(sigmas=grid, mu=.7)
        for sampler in ("heun", "er_sde"):
            scheduler = sampling.make_scheduler(template, sampler, "checkpoint", generator=torch.Generator().manual_seed(7))
            for _ in range(2):
                scheduler.set_timesteps(sigmas=grid, mu=.7)
                actual = scheduler.sigmas if sampler == "er_sde" else torch.cat([scheduler.sigmas[:1], scheduler.sigmas[1::2]])
                torch.testing.assert_close(actual[1:], euler.sigmas[1:])
                if sampler == "heun":
                    torch.testing.assert_close(actual, euler.sigmas)
                else:
                    self.assertLess(float(actual[0]), 1)
                sample = torch.ones(1, 4)
                for t in scheduler.timesteps:
                    sample = scheduler.step(torch.ones_like(sample) * .2, t, sample, return_dict=False)[0]
                self.assertTrue(torch.isfinite(sample).all())

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers"), "Install worker dependencies")
    def test_all_samplers_keep_bfloat16_latents_finite(self):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        sampling = self.sampling()
        for sampler in ("euler", "heun", "er_sde"):
            scheduler = sampling.make_scheduler(
                FlowMatchEulerDiscreteScheduler(), sampler, "simple",
                generator=torch.Generator().manual_seed(7),
            )
            scheduler.set_timesteps(sigmas=sampling.comfy_simple_sigmas(40), mu=.7)
            sample = torch.linspace(-1, 1, 12, dtype=torch.bfloat16).reshape(1, 3, 4)
            for timestep in scheduler.timesteps:
                velocity = sample * .15 + (timestep / 1000).to(sample.dtype).square() - .02
                sample = scheduler.step(velocity, timestep, sample, return_dict=False)[0]
                self.assertTrue(torch.isfinite(sample).all(), sampler)
                self.assertEqual(sample.dtype, torch.bfloat16)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("diffusers"), "Install worker dependencies")
    def test_sigmas_match_comfy_and_are_not_shifted_a_second_time(self):
        import math
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler

        # Comfy ModelSamplingFlux.set_parameters(shift=.69, timesteps=10000)
        t = torch.arange(1, 10001) / 10000
        table = math.exp(.69) / (math.exp(.69) + (1 / t - 1))
        # Comfy simple_scheduler selects from the descending sampling table.
        expected = torch.tensor([float(table[-(1 + int(i * len(table) / 40))]) for i in range(40)] + [0.])
        sampling = self.sampling()
        scheduler = sampling.base_scheduler(FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, shift_terminal=.02))
        scheduler.set_timesteps(sigmas=sampling.comfy_simple_sigmas(40), mu=.7)
        torch.testing.assert_close(scheduler.sigmas, expected, atol=1e-7, rtol=1e-6)
        self.assertEqual(len(scheduler.timesteps), 40)
        # Euler in Comfy: x_next = x + velocity * (sigma_next - sigma).
        sample = torch.ones(1, 4)
        velocity = torch.full_like(sample, .25)
        for index, timestep in enumerate(scheduler.timesteps):
            reference = sample + velocity * (expected[index + 1] - expected[index])
            sample = scheduler.step(velocity, timestep, sample, return_dict=False)[0]
            torch.testing.assert_close(sample, reference, atol=1e-6, rtol=1e-6)

    def test_simple_grid_has_forty_descending_nonzero_steps(self):
        sigmas = self.sampling().comfy_simple_sigmas(40)
        self.assertEqual(len(sigmas), 40)
        self.assertEqual(sigmas[0], 1)
        self.assertTrue(all(left > right for left, right in zip(sigmas, sigmas[1:])))
        self.assertGreater(sigmas[-1], 0)


if __name__ == "__main__":
    unittest.main()
