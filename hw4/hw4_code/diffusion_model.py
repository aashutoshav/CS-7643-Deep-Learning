import os
import torch
from randomizer import DL_random
from tqdm import tqdm
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from noise_prediction_net import ConditionalUnet1D
from noise_prediction_net import ConditionalUnet2D
from noise_prediction_net import TesterModel
from noise_scheduler import NoiseScheduler
from utils import get_best_device


class DiffusionModel:
    def __init__(
        self,
        input_shape,
        condition_dim,
        sequential=True,
        denoising_steps=100,
        p_uncond=0.2,
        weights_path=None,
        test=False,
    ):
        """
        Initialize the diffusion model
        Args:
            input_shape (tuple): the shape of the input data
            condition_dim (int): the dimension of the conditioning information
            sequential (bool): whether to use the sequential model (for robotics)
            denoising_steps (int): the number of denoising steps
            p_uncond (float): the probability of unconditional training
            weights_path (str): the path to load the weights from
            test (bool): whether to use the test model (only for unit tests and autograder, do not set this yourself)
        """
        # saves the number of training epochs that have been run
        self.training_epoch = 0
        self.input_shape = input_shape
        self.condition_dim = condition_dim
        self.sequential = sequential
        self.p_uncond = p_uncond
        self.denoising_steps = denoising_steps

        # initialize the noise scheduler with the number of denoising steps
        self.noise_scheduler = NoiseScheduler(self.denoising_steps)

        # initialize the noise prediction network
        if test:
            # Used for unit tests and autograder
            self.noise_pred_net = TesterModel(input_shape, condition_dim)
        elif self.sequential:
            # Used for robotics
            T, input_dim = input_shape
            self.noise_pred_net = ConditionalUnet1D(
                input_dim=input_dim, global_cond_dim=self.condition_dim
            )
        else:
            # Used for image generation
            c_in, H, W = input_shape
            assert H == W
            self.noise_pred_net = ConditionalUnet2D(
                input_dim=H, c_in=c_in, c_out=c_in, global_cond_dim=self.condition_dim
            )

        # initialize the device, ema model (helps performance, don't worry about calling this), and optimizer
        self.device = torch.device(get_best_device())
        self.ema = EMAModel(parameters=self.noise_pred_net.parameters(), power=0.75)
        # load weights if they are provided
        if weights_path is not None:
            self.noise_pred_net.load_state_dict(
                torch.load(weights_path, map_location=self.device, weights_only=True)
            )
            ema_path = weights_path.replace(".pth", "_ema.pth")
            self.ema.load_state_dict(
                torch.load(ema_path, map_location=self.device, weights_only=True)
            )
        self.noise_pred_net.to(self.device)
        self.ema.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.noise_pred_net.parameters(), lr=1e-4, weight_decay=1e-6
        )

    def train(self, data_loader, train_epochs=10):
        """
        train the diffusion model
        Args:
            data_loader (nn.Module): the diffusion model
            train_epochs (torch.utils.data.Dataset): the dataset
        """
        self.noise_pred_net.train()
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=len(data_loader) * train_epochs,
        )

        with tqdm(
            range(train_epochs), position=1, desc="Training Progress"
        ) as tdqm_epochs:
            for epoch in tdqm_epochs:
                average_loss = 0
                with tqdm(data_loader, position=0, desc="Batch") as tdqm_data_loader:
                    for data, cond in tdqm_data_loader:
                        data = data.to(self.device)
                        cond = cond.to(self.device)
                        loss = self.compute_loss_on_batch(data, cond)
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                        self.ema.step(self.noise_pred_net.parameters())
                        average_loss += loss.item()

                average_loss /= len(data_loader)
                tdqm_epochs.set_postfix(
                    {"Epoch": self.training_epoch, "Loss": average_loss}
                )
                self.training_epoch += 1

    def save_weights(self, path):
        """
        save the weights of the diffusion model to path
        save the weights of the ema model to path[:-4] + "_ema.pth"
        Args:
            path (str): the path to save the weights to
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.noise_pred_net.state_dict(), path)
        torch.save(self.ema.state_dict(), path[:-4] + "_ema.pth")

    def load_weights(self, path):
        """
        load the weights of the diffusion model from path
        load the weights of the ema model from path[:-4] + "_ema.pth"
        Args:
            path (str): the path to load the weights from
        """
        self.noise_pred_net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.ema.load_state_dict(
            torch.load(
                path[:-4] + "_ema.pth", map_location=self.device, weights_only=True
            )
        )
        self.noise_pred_net.to(self.device)
        self.ema.to(self.device)

    def compute_loss_on_batch(self, data, cond, seed=None):
        """
        ** Use DL_random() to generate any random numbers **

        train the diffusion model
        Args:
            cond (torch.Tensor): the conditioning information of shape (B, self.condition_dim)
            data (torch.Tensor): the data of shape (B, self.input_shape)
        Returns:
            loss (torch.Tensor): the training loss

        """
        #######################################################################
        # TODO: Implement unconditional training for classifier free guidance #
        #                                                                     #
        # Hint: DL_random(shape = (1,), normal = False, seed = seed).item()   #
        #       returns a random value between 0 and 1                        #
        # Hint: only a couple lines, 1 call to DL_random (don't forget seed)  #
        # Hint: This part is implemented in 2.2.2. Classifier Free Guidance   #
        #######################################################################
        ### BEGIN SOLUTION ###
        if DL_random(shape=(1,), normal=False, seed=seed).item() < self.p_uncond:
            cond = torch.zeros_like(cond)
        ### END SOLUTION ###

        loss = None

        #######################################################################
        # TODO: Implement loss computation                                    #
        # Hint: Use DL_random to generate the random noise                    #
        # Hint: Use DL_random with int_range to generate the timesteps        #
        # Hint: Don't forget seed when using DL_random                        #
        #######################################################################
        ### BEGIN SOLUTION ###
        batch_size = data.shape[0]

        timesteps = DL_random(
            shape=(batch_size,),
            int_range=(0, self.denoising_steps),
            seed=seed,
        ).long().to(data.device)

        noise = DL_random(shape=data.shape, seed=seed).to(data.device)

        noisy_data = self.noise_scheduler.add_noise(
            original_samples=data,
            noise=noise,
            timesteps=timesteps,
        )

        pred_noise = self.noise_pred_net(noisy_data, timesteps, cond)

        loss = torch.nn.functional.mse_loss(pred_noise, noise)
        ### END SOLUTION ###

        return loss

    def generate_sample(self, cond, guidance_weight=0.0, threshold=False, seed=None):
        """
        ** Use DL_random() to generate any random numbers **

        generate an output from the diffusion model
        Args:
            cond (torch.Tensor): the conditioning information of shape (B, self.condition_dim)
        Returns:
            sample (torch.Tensor): the generated sample of shape (B, self.input_shape)
        """
        # Store current model weights and copy EMA weights for inference
        self.ema.store(self.noise_pred_net.parameters())
        self.ema.copy_to(self.noise_pred_net.parameters())

        self.noise_pred_net.eval()
        sample = None

        with torch.no_grad():
            ###################################################################################
            # TODO: Generate a sample from the diffusion model using classifier free guidance #
            # Hint: Use DL_random to generate the initial noise (don't forget seed)           #
            # Hint: Use self.noise_pred_net (now has EMA weights loaded)                      #
            ###################################################################################
            ### BEGIN SOLUTION ###
            batch_size = cond.shape[0]
            sample_shape = (batch_size,) + tuple(self.input_shape)

            sample = DL_random(shape=sample_shape, seed=seed).to(self.device)

            uncond = torch.zeros_like(cond)

            for t in reversed(range(self.denoising_steps)):
                t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

                eps_cond = self.noise_pred_net(sample, t_tensor, cond)

                if guidance_weight > 0.0:
                    eps_uncond = self.noise_pred_net(sample, t_tensor, uncond)
                    pred_noise = (1.0 + guidance_weight) * eps_cond - guidance_weight * eps_uncond
                else:
                    pred_noise = eps_cond

                sample = self.noise_scheduler.denoise_step(
                    model_prediction=pred_noise,
                    t=t,
                    x_t=sample,
                    threshold=threshold,
                    seed=seed,
                )
            ### END SOLUTION ###

        # Restore original model weights
        self.ema.restore(self.noise_pred_net.parameters())
        self.noise_pred_net.train()
        return sample
