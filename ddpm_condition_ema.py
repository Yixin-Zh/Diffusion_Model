import copy
import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from model import UNet, UNet_Condition
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

import copy
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class DDPM_Condition:
    def __init__(self, noise_step=1000, init_beta=1e-4, end_beta=0.02, img_size=28, device='cuda'):

        self.noise_step = noise_step
        self.init_beta = init_beta
        self.end_beta = end_beta
        self.img_size = img_size
        self.device = device

        # Initialize the schedule
        self.beta = self.noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)


    def noise_schedule(self):
        schedule = torch.linspace(self.init_beta, self.end_beta, self.noise_step)
        return schedule

    def add_noise(self, x, t):
        """
        Add noise to the input image x
        arg: x shape (B, C, H, W)
        arg: t shape (B,)
        return: noised_x shape (B, C, H, W), noise shape (B, C, H, W)
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t].view(-1, 1, 1, 1))
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t].view(-1, 1, 1, 1))
        noise = torch.randn_like(x)

        noised_x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return noised_x, noise


    def sample_time_step(self, n):
        """
        uniform sample from {1,...,T}
        """
        return torch.randint(low=1, high=self.noise_step, size=(n,)).float()


    def sample(self, model, n, labels, cfg=3):
        """
        sample loop in DDPM
        args: model: the model to sample from
        args: n: number of samples
        """
        logging.info("Start sampling {} images".format(n))
        model.eval()
        with (torch.no_grad()):
            t = self.sample_time_step(n)
            x = torch.randn(n, 1, self.img_size, self.img_size).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_step))):
                t = (torch.ones(n)*i).long().to(self.device)

                predicted_noise = model(x, t, labels)
                if cfg > 0:
                    uncondition_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncondition_predicted_noise, predicted_noise, cfg)
                alpha = self.alpha[t].view(-1, 1, 1, 1)
                alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
                beta = self.beta[t].view(-1, 1, 1, 1)
                if i > 1:
                    z = torch.randn_like(x)
                else:
                    z = torch.zeros_like(x)
                x = 1./torch.sqrt(alpha) * (x - (1-alpha)/(torch.sqrt(1-alpha_hat))*predicted_noise) + torch.sqrt(beta)*z
        model.train()

        return x



def main():
    """
    train in DDPM
    """
    name = "DDPM_Condition_EMA"
    logging_setting(name)
    lr = 1e-3
    batch_size = 2
    epochs = 100
    dataloader, _ = get_data(batch_size, 28)
    model = UNet_Condition().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss = nn.MSELoss()
    logger = SummaryWriter(log_dir=os.path.join("logs", name))

    # Algo Logic: EMA
    ema = EMA(beta=0.999)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    ddpm = DDPM_Condition()
    # Algo Logic: Training Loop
    for epoch in range(epochs):
        for i, (x, labels) in enumerate(dataloader):
            x = x.cuda()
            labels = labels.cuda()
            t = ddpm.sample_time_step(x.size(0)).long().cuda()
            noised_x, noise = ddpm.add_noise(x, t)
            # 10% prob label == None
            if np.random.rand() < 0.1:
                labels = None
            optimizer.zero_grad()
            predicted_noise = model(noised_x, t, labels)
            loss_val = loss(predicted_noise, noise)
            loss_val.backward()
            optimizer.step()

            # step EMA
            ema.step_ema(ema_model, model)
            logger.add_scalar("Loss", loss_val.item(), epoch*len(dataloader)+i)
            if i % 10 == 0:
                logging.info("Epoch: {}, Iter: {}, Loss: {:.4f}".format(epoch, i, loss_val.item()))
                logger.add_images("Noised Image", noised_x, epoch*len(dataloader)+i)
                logger.add_images("Predicted Noise", predicted_noise, epoch*len(dataloader)+i)
                logger.add_images("Noise", noise, epoch*len(dataloader)+i)

        logger.add_scalar("Loss", loss_val.item(), epoch)
        if not os.path.exists(os.path.join("models", name)):
            os.makedirs(os.path.join("models", name))
        torch.save(model.state_dict(), os.path.join("models", name, f"model_epoch{epoch}.pth"))
        # save img
        if not os.path.exists(os.path.join("results", name)):
            os.makedirs(os.path.join("results", name))
        x = ddpm.sample(model, n=len(labels), labels=labels)
        torchvision.utils.save_image(x.cpu(), os.path.join("results", name, f"sample_epoch{epoch}.png"), nrow=8)
if __name__ == '__main__':
    main()