import numpy as np
import torch
import zarr


class PushTDataset:
    def __init__(
        self,
        dataset_path,
        batch_size=256,
        action_horizon=8,
        state_horizon=2,
        shuffle=True,
    ):
        self.batch_size = batch_size
        self.action_horizon = action_horizon
        self.state_horizon = state_horizon
        self.normalization_stats = {}
        self.shuffle = shuffle
        self.data_loader = self.load_dataset(dataset_path)

    # normalize data
    def get_data_stats(self, data, stats_key):
        """Get min and max of data"""
        data = data.reshape(-1, data.shape[-1])
        stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
        self.normalization_stats[stats_key] = stats

    def normalize_data(self, data, stats_key):
        """Normalize data to [-1, 1]"""
        stats = self.normalization_stats[stats_key]
        # nomalize to [0,1]
        ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
        return ndata

    def unnormalize_data(self, ndata, stats_key):
        """Unnormalize data from [-1, 1] to original range"""
        stats = self.normalization_stats[stats_key]
        ndata = (ndata + 1) / 2
        data = ndata * (stats["max"] - stats["min"]) + stats["min"]
        return data

    def load_dataset(self, dataset_path):
        """
        Construct the dataset and initialize the data loader

        The provided code sets up a loop over all episodes in the dataset,
        providing [min_start, max_start] variables to denote the range (inclusive)
        of indices relative to the episode's start_idx to slice windows of data
        from all_actions and all_states.

        Your task is to implement the logic to slice these windows, pad/reshape them
        to correct shape, and append them to final_actions and final_states.

        HINT: look at EnvRunner.init to understand the data format.

        Args:
            dataset_path (str): path to the dataset

        Returns:
            data_loader (torch.utils.data.DataLoader): data loader for the dataset
                - each batch is a tuple of (actions, states)
                - actions: (batch_size, diffusion model input shape)
                - states: (batch_size, diffusion model condition dim)
        """
        raw_data = zarr.open(dataset_path, mode="r")
        all_actions = raw_data["data"]["action"][:]
        all_states = raw_data["data"]["state"][:]
        episode_ends = raw_data["meta"]["episode_ends"][:]

        self.get_data_stats(all_actions, "action")
        self.get_data_stats(all_states, "state")

        all_actions = self.normalize_data(all_actions, "action")
        all_states = self.normalize_data(all_states, "state")

        all_actions = torch.from_numpy(all_actions)
        all_states = torch.from_numpy(all_states)

        final_actions = []
        final_states = []

        for i in range(len(episode_ends)):
            # Initializing some variables that may be useful for you
            start_idx = 0
            if i > 0:
                start_idx = episode_ends[i - 1]
            end_idx = episode_ends[i]
            episode_length = end_idx - start_idx
            min_start = -self.state_horizon + 1
            max_start = episode_length - 1

            for idx in range(min_start, max_start + 1):
                ### BEGIN SOLUTION ###
                state_indices = [
                    np.clip(idx - self.state_horizon + 1 + step, 0, episode_length - 1)
                    for step in range(self.state_horizon)
                ]
                state_window = all_states[start_idx + np.array(state_indices)].float().flatten()

                action_indices = [
                    np.clip(idx + step, 0, episode_length - 1)
                    for step in range(self.action_horizon)
                ]
                action_window = all_actions[start_idx + np.array(action_indices)].float()

                final_states.append(state_window)
                final_actions.append(action_window)
                ### END SOLUTION ###

        inputs = torch.stack(final_actions, dim=0)
        conds = torch.stack(final_states, dim=0)
        combined_dataset = torch.utils.data.TensorDataset(inputs, conds)
        return torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=True,
            num_workers=1,
            persistent_workers=True,
        )
