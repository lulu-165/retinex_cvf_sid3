import torch.nn.functional as F
import torch.nn as nn
import torch

mse = nn.MSELoss(reduction='mean')  # l2损失


# l1 = F.l1_loss()  # l1损失


def loss_aug(clean, clean1, noise_w, noise_w1, noise_b, noise_b1):
    loss1 = mse(clean1, clean)
    loss2 = mse(noise_w1, noise_w)
    loss3 = mse(noise_b1, noise_b)
    loss = loss1 + loss2 + loss3
    return loss


def loss_main(input_noisy, input_noisy_pred, clean, clean1, clean2, clean3, clean4, clean5, clean6, clean7,
              NiR, NiR1, NiR2, NiR3, NiR4, NiR5, NiR6, NiR7,
              NdR, NdR1, NdR2, NdR3, NdR4, NdR5, NdR6, NdR7,
              NiL, NiL1, NiL2, NiL3, NiL4, NiL5, NiL6, NiL7,
              NdL, NdL1, NdL2, NdL3, NdL4, NdL5, NdL6, NdL7,
              Nt, Nt1, Nt2, Nt3, Nt4, Nt5, Nt6, Nt7, R, L, max_rgb,
              std, gamma):
    loss1 = mse(input_noisy_pred, input_noisy)

    loss2 = mse(clean1, clean)

    loss3 = mse(clean2, clean)
    loss4 = mse(Nt3, Nt)
    loss5 = mse(clean4, clean)
    loss6 = mse(NdR4, NdR)
    loss7 = mse(Nt4, Nt)
    loss8 = mse(NdR5, NdR)
    loss9 = mse(NdR6, NdR)
    loss10 = mse(Nt6, Nt)
    loss11 = mse(clean7, clean)

    # loss13 = mse(NdR, torch.zeros_like(NdR))\
    loss12 = mse(NiR, torch.zeros_like(NiR))
    loss13 = mse(NdL, torch.zeros_like(NdL))
    # loss12 = mse(NiL, torch.zeros_like(NiL))
    loss14 = mse(NdR1, torch.zeros_like(NdR1))
    loss15 = mse(Nt1, torch.zeros_like(Nt1))
    loss16 = mse(Nt2, torch.zeros_like(Nt2))
    loss17 = mse(clean3, torch.zeros_like(clean3))
    loss18 = mse(NdR7, torch.zeros_like(NdR7))

    sigma_Nt = torch.std(Nt.reshape([Nt.shape[0], Nt.shape[1], -1]), -1)
    sigma_NdR = torch.std(NdR.reshape([NdR.shape[0], NdR.shape[1], -1]), -1)
    blur_clean = F.avg_pool2d(clean, kernel_size=6, stride=1, padding=3)
    clean_mean = torch.mean(torch.square(torch.pow(blur_clean, gamma).reshape([clean.shape[0], clean.shape[1], -1])),
                            -1)  # .detach()
    sigma_NtNdR = torch.sqrt(
        clean_mean * torch.square(sigma_NdR) + torch.square(sigma_Nt) + torch.square(sigma_Nt) * torch.square(
            sigma_NdR))
    loss19 = mse(sigma_NtNdR, std)

    loss20 = F.l1_loss(L * R, input_noisy)

    # # 平滑损失
    loss21 = smooth(L, R)
    loss22 = F.l1_loss(L, max_rgb)

    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16 + loss17 + loss18 + loss20 + .1 * loss21 + loss22

    return loss


def smooth(input_L, input_R):
    input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
    input_R = torch.unsqueeze(input_R, dim=1)
    input_L = input_L.mean(dim=1, keepdim=True)
    x = 1
    y = 2
    return torch.mean(gradient(input_L, x) * torch.exp(-10 * ave_gradient(input_R, x)) +
                      gradient(input_L, y) * torch.exp(-10 * ave_gradient(input_R, y)))


def gradient(input_tensor, direction):
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
    smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)

    if direction == 1:
        kernel = smooth_kernel_x
    elif direction == 2:
        kernel = smooth_kernel_y
    grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                  stride=1, padding=1))
    return grad_out


def ave_gradient(input_tensor, direction):
    return F.avg_pool2d(gradient(input_tensor, direction),
                        kernel_size=3, stride=1, padding=1)


if __name__ == '__main__':
    print('loss')
