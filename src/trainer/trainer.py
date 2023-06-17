import numpy as np
import torch
from src.base import BaseTrainer
from src.utils import inf_loop, MetricTracker
from src.model.metric import psnr
from torchvision.utils import save_image
import torch.nn.functional as F
from torch import autograd
import os

patch_size = 256


def sgd(weight: torch.Tensor, grad: torch.Tensor, meta_lr) -> torch.Tensor:
    weight = weight - meta_lr * grad
    return weight


def padr(img):
    pad = 20
    pad_mod = 'reflect'
    img_pad = F.pad(input=img, pad=(pad, pad, pad, pad), mode=pad_mod)
    return img_pad


def padr_crop(img):
    pad = 20
    pad_mod = 'reflect'
    img = F.pad(input=img[:, :, pad:-pad, pad:-pad], pad=(pad, pad, pad, pad), mode=pad_mod)
    return img


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, test_data_loader,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer=optimizer, config=config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.test_data_loader = test_data_loader
        self.do_test = self.test_data_loader is not None
        self.do_test = True
        self.gamma = 1.0
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('Total_loss', writer=self.writer)
        self.test_metrics = MetricTracker('psnr', 'ssim', writer=self.writer)
        if os.path.isdir('../output') == False:
            os.makedirs('../output/')
        if os.path.isdir('../output/C') == False:
            os.makedirs('../output/C/')
        if os.path.isdir('../output/GT') == False:
            os.makedirs('../output/GT/')
        if os.path.isdir('../output/N_i') == False:
            os.makedirs('../output/N_i/')
        if os.path.isdir('../output/N_d') == False:
            os.makedirs('../output/N_d/')
        if os.path.isdir('../output/I') == False:
            os.makedirs('../output/I/')

    def _train_epoch(self, epoch):

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (target, input_noisy, input_GT, max_rgb, std) in enumerate(self.data_loader):
            max_rgb = torch.cat([max_rgb, max_rgb, max_rgb], dim=3)
            max_rgb = np.transpose(max_rgb, (0, 3, 1, 2))
            input_noisy = input_noisy.to(self.device)
            input_GT = input_GT.to(self.device)
            max_rgb = max_rgb.to(self.device)
            std = std.to(self.device)
            pad = 20
            input_noisy = padr(input_noisy)
            input_GT = padr(input_GT)
            # max_rgb = max_rgb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).permute(1, 0, 2, 3)

            max_rgb = padr(max_rgb)
            # max_rgb = max_rgb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).permute(1, 0, 2, 3)
            self.optimizer.zero_grad()

            clean, NiR, NdR, NiL, NdL, Nt, R, L, IR, IL = self.model(input_noisy)
            clean1, NiR1, NdR1, NiL1, NdL1, Nt1, R1, L1, IR1, IL1 = self.model(padr_crop(clean))
            clean2, NiR2, NdR2, NiL2, NdL2, Nt2, R2, L2, IR2, IL2 = self.model(
                padr_crop((clean + torch.pow(clean, self.gamma) * NdR)))  # 1
            clean3, NiR3, NdR3, NiL3, NdL3, Nt3, R3, L3, IR3, IL3 = self.model(padr_crop(Nt + Nt * NdR))
            clean4, NiR4, NdR4, NiL4, NdL4, Nt4, R4, L4, IR4, IL4 = self.model(
                padr_crop((clean + torch.pow(clean, self.gamma) * NdR) + Nt + Nt * NdR))  # 1

            clean5, NiR5, NdR5, NiL5, NdL5, Nt5, R5, L5, IR5, IL5 = self.model(padr_crop(NdR * clean + Nt * NdR))
            clean6, NiR6, NdR6, NiL6, NdL6, Nt6, R6, L6, IR6, IL6 = self.model(padr_crop(Nt + NdR * clean + Nt * NdR))
            clean7, NiR7, NdR7, NiL7, NdL7, Nt7, R7, L7, IR7, IL7 = self.model(padr_crop((clean + Nt)))  # 6

            input_noisy_pred = clean + torch.pow(clean, self.gamma) * NdR + Nt + Nt * NdR

            loss = self.criterion[0](input_noisy, input_noisy_pred, clean, clean1, clean2, clean3, clean4, clean5,
                                     clean6, clean7,
                                     NiR, NiR1, NiR2, NiR3, NiR4, NiR5, NiR6, NiR7,
                                     NdR, NdR1, NdR2, NdR3, NdR4, NdR5, NdR6, NdR7,
                                     NiL, NiL1, NiL2, NiL3, NiL4, NiL5, NiL6, NiL7,
                                     NdL, NdL1, NdL2, NdL3, NdL4, NdL5, NdL6, NdL7,
                                     Nt, Nt1, Nt2, Nt3, Nt4, Nt5, Nt6, Nt7, R, L,max_rgb,
                                     std, self.gamma)
            loss_total = loss
            loss_total.backward()

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} TotalLoss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_total.item()
                ))

            if batch_idx == self.len_epoch:
                break

            del target
            del loss_total

        log = self.train_metrics.result()

        if self.do_test:
            if epoch > 100 or epoch % 1 == 0:
                test_log = self._test_epoch(epoch, save=False)
                log.update(**{'test_' + k: v for k, v in test_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.writer.close()

        return log

    def _test_epoch(self, epoch, save=False):

        self.test_metrics.reset()

        # with torch.no_grad():
        if save == True:
            os.makedirs('../output/C/' + str(epoch))
            os.makedirs('../output/N_d/' + str(epoch))
            os.makedirs('../output/N_i/' + str(epoch))
        for batch_idx, (target, input_noisy, input_GT, max_rgb,std) in enumerate(self.test_data_loader):
            max_rgb = torch.cat([max_rgb, max_rgb, max_rgb], dim=3)
            max_rgb = np.transpose(max_rgb, (0, 3, 1, 2))
            input_noisy = input_noisy.to(self.device)
            input_GT = input_GT.to(self.device)
            max_rgb = max_rgb.to(self.device)
            pad = 20
            input_noisy = padr(input_noisy)
            input_GT = padr(input_GT)
            max_rgb = padr(max_rgb)

            clean, NiR, NdR, NiL, NdL, Nt, R, L, IR, IL = self.model(input_noisy)

            size = [Nt.shape[0], Nt.shape[1], Nt.shape[2] * Nt.shape[3]]
            noise_b_normal = (Nt - torch.min(Nt.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1)) / (
                    torch.max(Nt.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1) -
                    torch.min(Nt.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1))
            noise_w_normal = (NdR - torch.min(NdR.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1)) / (
                    torch.max(NdR.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1) -
                    torch.min(NdR.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1))
            if save == True:
                for i in range(input_noisy.shape[0]):
                    save_image(torch.clamp(clean[i, :, pad:-pad, pad:-pad], min=0, max=1).detach().cpu(),
                               '../output/C/' + str(epoch) + '/' + target['dir_idx'][i] + '.PNG')
                    save_image(torch.clamp(input_GT[i, :, pad:-pad, pad:-pad], min=0, max=1).detach().cpu(),
                               '../output/GT/' + target['dir_idx'][i] + '.PNG')
                    save_image(torch.clamp(noise_b_normal[i, :, pad:-pad, pad:-pad], min=0, max=1).detach().cpu(),
                               '../output/N_i/' + str(epoch) + '/' + target['dir_idx'][i] + '.PNG')
                    save_image(torch.clamp(noise_w_normal[i, :, pad:-pad, pad:-pad], min=0, max=1).detach().cpu(),
                               '../output/N_d/' + str(epoch) + '/' + target['dir_idx'][i] + '.PNG')
                    save_image(torch.clamp(input_noisy[i, :, pad:-pad, pad:-pad], min=0, max=1).detach().cpu(),
                               '../output/I/' + target['dir_idx'][i] + '.PNG')

            self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'test')
            for met in self.metric_ftns:
                if met.__name__ == "psnr":
                    psnr = met(input_GT[:, :, pad:-pad, pad:-pad].to(self.device),
                               torch.clamp(clean[:, :, pad:-pad, pad:-pad], min=0, max=1))
                    self.test_metrics.update('psnr', psnr)
                elif met.__name__ == "ssim":
                    self.test_metrics.update('ssim', met(input_GT[:, :, pad:-pad, pad:-pad].to(self.device),
                                                         torch.clamp(clean[:, :, pad:-pad, pad:-pad], min=0, max=1)))
            self.writer.close()

            del target

        self.writer.close()
        return self.test_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
