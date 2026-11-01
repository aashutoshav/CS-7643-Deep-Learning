from .environment import PushTEnv
from .dataset import PushTDataset
from diffusion_model import DiffusionModel
import torch
from collections import deque
import numpy as np
from tqdm import tqdm
from utils import get_best_device


class EnvRunner:
    def __init__(self):
        # Initialize parameters (do not change these)
        self.action_dim = 2
        self.state_dim = 5
        self.max_timesteps = 250
        self.device = torch.device(get_best_device())

        # Define H_a, H_s and num_actions (you can vary these during your experiments)
        self.action_horizon = 16
        self.state_horizon = 2
        self.num_actions = 9

        self.input_shape = (self.action_horizon, self.action_dim)
        self.cond_dim = self.state_dim * self.state_horizon

        self.policy = DiffusionModel(
            input_shape=self.input_shape,
            condition_dim=self.cond_dim,
            sequential=True,
            denoising_steps=100,
        )

    def load_data(self, dataset_path, batch_size=256):
        data = PushTDataset(
            dataset_path=dataset_path,
            action_horizon=self.action_horizon,
            state_horizon=self.state_horizon,
        )
        self.data = data

    def train_policy(self, epochs=10):
        self.policy.train(self.data.data_loader, epochs)

    @torch.no_grad()
    def run_rollout(self):
        """
        Run a rollout in the environment using the trained policy.

        Action Generation:
        - Use self.state_deque to store the past states
        - Use self.action_deque to store the generated actions
        - Generate actions only when the action deque is empty.
        - A correct implementation wille execute self.num_actions actions before generating new ones.

        Normalization
        - self.data.normalize_data(state(s), 'state') to normalize states passed to the policy
        - self.data.unnormalize_data(action(s), 'action') to unnormalize generated actions

        :returns: imgs (list): list of images from the rollout
                  max_images (list): list of images ending at the timestep with max reward
        """
        env = PushTEnv()
        env.seed(100000)
        initial_state = env.reset()

        # initialize action and state deques
        state_deque = deque([initial_state] * self.state_horizon, maxlen=self.state_horizon)
        action_deque = deque(maxlen=self.num_actions)

        step_idx = 0
        imgs = [env.render(mode="rgb_array")]
        done = False

        max_index = None
        max_reward = -1

        with tqdm(total=self.max_timesteps, desc="Eval PushTStateEnv") as pbar:
            # loop until max timesteps reached or done (block pushing is successful)
            while not done:
                # Generate and add self.num_actions actions to the action deque if it is empty
                ### BEGIN SOLUTION ###
                if len(action_deque) == 0:
                    raw_states = np.array(state_deque)
                    norm_states = self.data.normalize_data(raw_states, "state")
                    
                    cond = torch.from_numpy(norm_states).float().flatten().unsqueeze(0).to(self.device)

                    pred_actions = self.policy.generate_sample(cond)

                    pred_actions = pred_actions.detach().cpu().numpy().squeeze(0)  # Shape: (action_horizon, action_dim)
                    unnorm_actions = self.data.unnormalize_data(pred_actions, "action")

                    action_deque.extend(unnorm_actions[:self.num_actions])

                action = action_deque.popleft()
                ### END SOLUTION ###

                # take one action step in the environment, get the resulting state, reward, and done
                state, reward, done, _ = env.step(action)
                state_deque.append(state)
                imgs.append(env.render(mode="rgb_array"))
                if reward > max_reward:
                    max_reward = reward
                    max_index = step_idx
                # stop if max timesteps reached
                if step_idx > self.max_timesteps:
                    done = True
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)

        if max_reward <= 0:
            max_images = imgs[-10:]
        else:
            max_images = imgs[max_index - 10 : max_index + 1]
        print("Max reward: {}".format(max_reward))
        return imgs, max_images
