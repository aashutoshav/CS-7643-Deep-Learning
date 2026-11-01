import unittest
from noise_scheduler import NoiseScheduler
from diffusion_model import DiffusionModel
from robotics.dataset import PushTDataset
from randomizer import DL_random
from utils import get_best_device

import torch
import numpy as np
import os

# Global relative tolerance for testing
RTOL = 1e-2


def set_seed(seed: int = 42) -> None:
    # np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


class TestCases(unittest.TestCase):
    def test_noise_scheduler_init(self):
        scheduler_100 = NoiseScheduler(100)
        scheduler_50 = NoiseScheduler(50)
        alpha_bar_100 = np.load("test_resources/scheduler_init_alpha_bar_100.npy")
        alpha_bar_50 = np.load("test_resources/scheduler_init_alpha_bar_50.npy")

        assert len(scheduler_100.alphas) == 100
        assert len(scheduler_100.alpha_bars) == 100
        np.testing.assert_allclose(
            actual=scheduler_100.alphas[0].item(), desired=0.9999, rtol=RTOL
        )
        np.testing.assert_allclose(
            actual=scheduler_100.alpha_bars[-1].item(), desired=alpha_bar_100, rtol=RTOL
        )
        assert len(scheduler_50.alphas) == 50
        assert len(scheduler_50.alpha_bars) == 50
        np.testing.assert_allclose(
            actual=scheduler_50.alphas[0].item(), desired=0.9999, rtol=RTOL
        )
        np.testing.assert_allclose(
            actual=scheduler_50.alpha_bars[-1].item(), desired=alpha_bar_50, rtol=RTOL
        )

        print("NoiseScheduler() initialization test case passed!")

    def test_robotics_dataset(self):
        dataset = PushTDataset(
            "robotics/push_t_dataset.zarr",
            batch_size=256,
            action_horizon=8,
            state_horizon=2,
        )
        data_loader = dataset.data_loader
        data_point = next(iter(data_loader))
        assert len(data_loader) == 101
        assert data_point[0].shape == torch.Size([256, 8, 2])
        assert data_point[1].shape == torch.Size([256, 10])

        print("PushTDataset() test case passed!")

    def test_add_noise(self):
        set_seed(42)
        device = torch.device(get_best_device())
        test_data = torch.ones((5, 30)).to(device)

        test_noise = DL_random(shape=(5, 30), seed=42).to(device)  # 0-1 Uniform Noise
        test_time = torch.tensor([10, 30, 60, 80, 90]).to(device)

        device = "cpu"
        ground_truth = np.load("test_resources/add_noise.npy")
        output = (
            NoiseScheduler(100)
            .add_noise(test_data, test_noise, test_time)
            .to(device)
            .numpy()
        )
        np.testing.assert_allclose(actual=output, desired=ground_truth, rtol=RTOL)

        print("add_noise() test case passed!")

    def test_denoise_step(self):
        set_seed(42)
        device = torch.device(get_best_device())
        test_model_prediction = DL_random(shape=(5, 30), seed=42).to(device)
        test_t = 56
        test_x_t = DL_random(shape=(5, 30), seed=42).to(device)

        device = "cpu"
        ground_truth = np.load("test_resources/denoise_step.npy")
        output = (
            NoiseScheduler(100)
            .denoise_step(test_model_prediction, test_t, test_x_t, seed=42)
            .to(device)
            .numpy()
        )

        np.testing.assert_allclose(actual=output, desired=ground_truth, rtol=RTOL)
        print("denoise_step() test case passed!")

    def test_threshold_denoise_step(self):
        set_seed(42)
        device = torch.device(get_best_device())
        test_model_prediction = DL_random(shape=(5, 30), seed=42).to(device) * 3
        test_t = 56
        test_x_t = DL_random(shape=(5, 30), seed=42).to(device) * 3

        device = "cpu"
        output = (
            NoiseScheduler(100)
            .denoise_step(
                test_model_prediction, test_t, test_x_t, threshold=True, seed=42
            )
            .to(device)
            .numpy()
        )
        ground_truth = np.load("test_resources/denoise_step_threshold.npy")

        np.testing.assert_allclose(actual=output, desired=ground_truth, rtol=RTOL)
        print("threshold_denoise_step() test case passed!")

    def test_compute_loss_on_batch(self):
        set_seed(42)
        device = torch.device(get_best_device())
        model = DiffusionModel(
            (5, 10),
            condition_dim=5,
            p_uncond=0,
            test=True,
            weights_path="test_resources/test_model_weights.pth",
        )

        test_data = DL_random(shape=(6, 5, 10), seed=42).to(device)
        test_cond = DL_random(shape=(6, 5), seed=42).to(device)

        device = "cpu"
        output = (
            model.compute_loss_on_batch(test_data, test_cond, seed=42)
            .to(device)
            .detach()
            .numpy()
        )

        ground_truth = 1.6004212

        np.testing.assert_allclose(actual=output, desired=ground_truth, rtol=RTOL)

        print("compute_loss_on_batch() test case passed!")

    def test_compute_loss_with_cfg(self):
        set_seed(42)
        device = torch.device(get_best_device())
        model = DiffusionModel(
            (5, 10),
            condition_dim=5,
            p_uncond=1,
            test=True,
            weights_path="test_resources/test_model_weights.pth",
        )

        test_data = DL_random(shape=(6, 5, 10), seed=42).to(device)
        test_cond = DL_random(shape=(6, 5), seed=42).to(device)

        device = "cpu"

        output = (
            model.compute_loss_on_batch(test_data, test_cond, seed=42)
            .to(device)
            .detach()
            .numpy()
        )
        ground_truth = 1.5380142

        np.testing.assert_allclose(actual=output, desired=ground_truth, rtol=RTOL)

        print("compute_loss_with_cfg() test case passed!")

    def test_generate_sample(self):
        set_seed(42)
        device = torch.device(get_best_device())
        model = DiffusionModel(
            (5, 10),
            condition_dim=5,
            p_uncond=1,
            test=True,
            weights_path="test_resources/test_model_weights.pth",
        )

        test_cond = DL_random(shape=(6, 5), seed=42).to(device)

        device = "cpu"

        desired = np.load("test_resources/generate_sample.npy")
        sample = model.generate_sample(test_cond, seed=42).to(device).numpy()
        np.testing.assert_allclose(actual=sample, desired=desired, rtol=RTOL)

        print("generate_sample_step() test case passed!")

        guidance = 2.0
        sample_wguidance = (
            model.generate_sample(test_cond, guidance_weight=guidance, seed=42)
            .to(device)
            .numpy()
        )

        desired = np.load("test_resources/generate_sample_guidance.npy")
        np.testing.assert_allclose(actual=sample_wguidance, desired=desired, rtol=RTOL)

        print("generate_sample_step() with guidance test case passed!")


if __name__ == "__main__":
    unittest.main()
