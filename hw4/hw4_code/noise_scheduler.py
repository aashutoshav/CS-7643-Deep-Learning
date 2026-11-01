from randomizer import DL_random
import numpy as np
import torch
from utils import get_best_device


class NoiseScheduler:
    def __init__(self, num_steps, beta_start=0.0001, beta_end=0.02):
        # initialize the beta parameters (variance) of the scheduler
        self.beta_start = beta_start
        self.beta_end = beta_end

        # number of inference steps (same as num training steps)
        self.num_steps = num_steps

        # linear schedule for beta
        self.betas = np.linspace(self.beta_start, self.beta_end, self.num_steps)

        ###########################################################
        # TODO: Compute alphas and alpha_bars (refer to DDPM paper)
        # Hint: Use np.cumprod when initializing alpha_bars       #
        ###########################################################
        ### BEGIN SOLUTION ###
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

        ### END SOLUTION ###

        # convert to tensors
        self.device = torch.device(get_best_device())
        self.alphas = torch.from_numpy(self.alphas).float().to(self.device)
        self.alpha_bars = torch.from_numpy(self.alpha_bars).float().to(self.device)
        self.betas = torch.from_numpy(self.betas).float().to(self.device)

    def denoise_step(self, model_prediction, t, x_t, threshold=False, seed=None):
        """
        ** Use DL_random() to generate any random numbers **

        Implement a step of the reverse denoising process
        Args:
            model_prediction (torch.Tensor): the output of the noise prediction model (B, input_shape)
            t (int): the current timestep
            x_t (torch.Tensor): the previous timestep (B, input_shape)
            threshold (bool): whether to threshold x_0, implemented in part 2.3
        Returns:
            x_t_prev (torch.Tensor): the denoised previous timestep (B, input_shape)

        """

        x_t_prev = None
        if not threshold:
            ####################################################
            # TODO: Implement a denoising step                 #
            # Hint: 1 call to DL_random (don't forget seed)    #
            # NOTE: Use zero indexing, ie. check for t > 0     #
            # NOTE: Use beta instead of 1-alpha                #
            # NOTE: Use plain beta for the variance instead of #
            #       beta tilde, see 3.2 in DDPM Paper          #
            ####################################################
            ### BEGIN SOLUTION ###
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]

            mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * model_prediction)

            if t > 0:
                z = DL_random(x_t.shape, seed=seed).to(x_t.device)
                x_t_prev = mean + torch.sqrt(beta_t) * z
            else:
                x_t_prev = mean
            ### END SOLUTION ###

        else:
            ######################################################
            # TODO: Implement a denoising step with thresholding #
            # Hint: The main difference is how you compute       #
            #       the mean of the x_t_prev                     #
            # Hint: Clamp between -1 and 1                       #
            # Hint: 1 call to DL_random (don't forget seed)      #
            # NOTE: Use zero indexing, ie. check for t > 0       #
            # NOTE: Compute the coefficients of x_0 and x_t      #
            #       in their own variables                       #
            # NOTE: Use beta instead of 1-alpha                  #
            # NOTE: Use plain beta for the variance instead of   #
            #       beta tilde.                                  #
            ######################################################
            ### BEGIN SOLUTION ###
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]
            alpha_bar_prev = self.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=x_t.device)

            pred_x_0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * model_prediction) / torch.sqrt(alpha_bar_t)
            pred_x_0 = torch.clamp(pred_x_0, -1.0, 1.0)
            
            coef_x0 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1.0 - alpha_bar_t)
            coef_xt = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t)

            mean = coef_x0 * pred_x_0 + coef_xt * x_t

            if t > 0:
                z = DL_random(x_t.shape, seed=seed).to(x_t.device)
                x_t_prev = mean + torch.sqrt(beta_t) * z
            else:
                x_t_prev = mean
            ### END SOLUTION ###

        return x_t_prev

    def add_noise(self, original_samples, noise, timesteps):
        """
        add noise to the original samples - the forward diffusion process.

        NOTE: the input_shape of original_samples may vary in the number of dimensions,
            so you will need to address any shape mismatches that arise

        Args:
            original_samples (torch.Tensor): the uncorrupted original samples (B, input_shape)
            noise (torch.Tensor): random gaussian noise (B, input_shape)
            timesteps (torch.Tensor): the timesteps for noise addition (B,)
        Returns:
            noisy_samples (torch.Tensor): corrupted samples with amount of noise added based
                                          on the corresponding timestep (B, input_shape)
        """
        noisy_samples = None
        ###########################################
        # TODO: Implement forward noising process #
        ###########################################
        ### BEGIN SOLUTION ###
        alpha_bars = self.alpha_bars[timesteps]

        dims_to_add = original_samples.ndim - alpha_bars.ndim
        view_shape = alpha_bars.shape + (1,) * dims_to_add
        alpha_bars = alpha_bars.view(view_shape)

        noisy_samples = torch.sqrt(alpha_bars) * original_samples + torch.sqrt(1.0 - alpha_bars) * noise
        ### END SOLUTION ###

        return noisy_samples
