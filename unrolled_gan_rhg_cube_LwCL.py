import os
import numpy as np
import matplotlib.pyplot as plt
import copy

from tqdm import tqdm_notebook
# import fid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io
import argparse
import time
import seaborn
import csv
import higher
import random
import hypergrad as hg
from mixture_gaussian_cube import data_generator

plt.style.use('ggplot')
parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

parser.add_argument('--method', default='GAN', type=str,
                    help='')
parser.add_argument('--WGAN', default='1', type=str,
                    help='')
parser.add_argument('--act', default='tanh', type=str,
                    help='')
parser.add_argument('--noise', default=0.01, type=float,
                    help='')
parser.add_argument('--n', default=3, type=int,
                    help='')
parser.add_argument('--std', default=0.02, type=float,
                    help='')
parser.add_argument('--us', default=10, type=int,
                    help='')
parser.add_argument('--num_iter', default=200000, type=int,
                    help='')
parser.add_argument('--log_iter', default=200, type=int,
                    help='')
parser.add_argument('--dlr', default=1e-3, type=float,
                    help='')
parser.add_argument('--zlr', default=1e-3, type=float,
                    help='')
parser.add_argument('--glr', default=1e-5, type=float,
                    help='')
parser.add_argument('--GN', action='store_true',default=False,
                    help='')
parser.add_argument('--c', default=0.05, type=float,
                    help='')
parser.add_argument('--GP', action='store_true', default=False,
                    help='')
parser.add_argument('--lamda', default=0.005, type=float,
                    help='')
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# device='cpu'
device='cuda'
if torch.cuda.is_available():
    cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
else:
    cuda = False
alphaF = torch.tensor(1).requires_grad_(False)
alphaR = torch.tensor(0).requires_grad_(False)
args.alphaF=torch.tensor(1).requires_grad_(False)
args.alphaR=torch.tensor(0).requires_grad_(False)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(30)



def copy_parameter(y, z):
    # print(loss_L1(y.parameters()))
    # print(loss_L1(z.parameters()))
    for p, q in zip(y.parameters(), z.parameters()):
        p.data = q.clone().detach().requires_grad_()
    # print(loss_L1(y.parameters()))
    # print('-'*80)
    return y


def copy_parameter_from_list(y, z):
    # print(loss_L1(y.parameters()))
    # print(loss_L1(z.parameters()))
    for p, q in zip(y.parameters(), z):
        p.data = q.clone().detach().requires_grad_()
    # print(loss_L1(y.parameters()))
    # print('-'*80)
    return y

def frnp(x):
    t = torch.from_numpy(x).to(device)
    return t


def tonp(x):
    # print(x)
    return x.detach().cpu().numpy()



def loss_L2(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 2) ** 2
    return loss


def loss_2L2(parameters1, parameters2):
    loss = 0
    for w1, w2 in zip(parameters1, parameters2):
        loss += torch.norm(w1 - w2, 2) ** 2
    return loss


def loss_L2(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 2) ** 2
    return loss


def loss_2L2(parameters1, parameters2):
    loss = 0
    for w1, w2 in zip(parameters1, parameters2):
        loss += torch.norm(w1 - w2, 2) ** 2
    return loss


def add_sn(m):
    for name, layer in m.named_children():
        m.add_module(name, add_sn(layer))
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(m)
    else:
        return m


def Binarization(x):
    x_bi = torch.zeros_like(x)
    xc = x.to('cpu')
    for i in range(x.shape[0]):
        # print(x)
        if x[i] >= 0.5:
            x_bi[i] = 1
        else:
            x_bi[i] = 0
    return x_bi


def D_acc(D, sample, gsample):
    # D = D.cpu()
    # sample = sample.cpu()
    drd = D(sample).detach()
    dfd = D(gsample).detach()
    d_real_decision = Binarization(drd)
    d_fake_decision = Binarization(dfd)
    drr = sum(d_real_decision)
    drf = len(d_real_decision) - sum(d_real_decision)
    dfr = sum(d_fake_decision)
    dff = len(d_fake_decision) - sum(d_fake_decision)
    p = drr / (drr + dfr + 1e-10)
    r = drr / (drr + drf + 1e-10)
    F1 = 2 * (p * r) / (p + r)
    return p, r, F1


def Exponential_moving_average(alpha, v, gamma=0.99):
    return torch.mean(gamma * alpha + (1 - alpha) * v)



## choose uniform mixture gaussian or weighted mixture gaussian
dset = data_generator(n=args.n, std=args.std)
# dset.random_distribution()
dset.uniform_distribution()


# plt.plot(dset.p)
# plt.title('Weight of each gaussian')
# plt.show()
# plt.close()


def plot(points, title):
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', alpha=0.5)
    plt.scatter(dset.centers[:, 0], dset.centers[:, 1], s=100, c='g', alpha=0.5)
    plt.title(title)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.show()
    plt.close()


log_interval = 0
# unrolled_steps=0
prefix = 'target'
BIM = False
WGAN = False
GN = False
prox = False


# plot the samples through iterations
def plot_target(samples):
    xmax = 15
    cols = len(samples)
    bg_color = seaborn.color_palette('Greens', n_colors=256)[0]
    plt.figure(figsize=(2 * cols, 2))
    for i, samps in enumerate(samples):
        if i == 0:
            ax = plt.subplot(1, cols, 1)
        else:
            plt.subplot(1, cols, i + 1, sharex=ax, sharey=ax)
        ax2 = seaborn.kdeplot(x=samps[:, 0], y=samps[:, 1], fill=True, cmap='Greens', n_levels=20,
                              clip=[[-xmax, xmax]] * 2)
        plt.xticks([])
        plt.yticks([])
        plt.title('step %d' % (i * log_interval))

    plt.gcf().tight_layout()
    plt.savefig('D:/Code_2023/GDC/LwCL_results_cube/' + prefix + '_' + args.method + 'target' + '.png')
    plt.show()
    plt.close()


def plot_samples(samples, glist=None, KLlist=None, JSlist=None,step=0):
    xmax = 15
    cols = len(samples)
    bg_color = seaborn.color_palette('Greens', n_colors=256)[0]
    plt.figure(figsize=(2 * cols, 2))
    for i, samps in enumerate(samples):
        if i == 0:
            ax = plt.subplot(1, cols, 1)
        else:
            plt.subplot(1, cols, i + 1, sharex=ax, sharey=ax)
        ax2 = seaborn.kdeplot(x=samps[:, 0], y=samps[:, 1], fill=True, cmap='Greens', n_levels=50,clip=[[-xmax, xmax]] * 2
                              )
        plt.xticks([])
        plt.yticks([])
        plt.title('step %d' % (step))

    if BIM:
        ax.set_ylabel('Ours')
    else:
        ax.set_ylabel('%d unrolling steps' % unrolled_steps)
    plt.gcf().tight_layout()
    # plt.savefig('Cube_results/'+ prefix + 'm{}_step{}'.format(args.method,step) + '.png')# time.strftime("%Y_%m_%d_%H_%M_%S")) + '.png')

    plt.savefig('D:/Code_2023/GDC/LwCL_results_cube/' + prefix + 'm{}_step{}'.format(args.method, step) + '.png')
    if glist != None:
        log_path = "{}_{}.csv".format(args.method, time.strftime("%Y_%m_%d_%H_%M_%S"))
        with open(log_path, 'a', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            # csv_writer.writerow([args])
            csv_writer.writerow(['g loss', glist])
            csv_writer.writerow(['KL', KLlist])
            csv_writer.writerow(['JS', JSlist])
    # plt.show()
    plt.close()


sample_points0 = dset.sample(512)
sample_points = sample_points0 + args.noise * np.random.randn(512, 3)
# scipy.io.savemat(
#     'Cube_results/target_cube_m{}_n{}_std{}.mat'.format(args.method,args.n, args.std),
#     mdict={'data': sample_points0})
scipy.io.savemat(
    'D:/Code_2023/GDC/LwCL_results_cube/target_m{}_n{}_std{}.mat'.format(args.method,args.n, args.std),
    mdict={'data': sample_points0})
plot_target([sample_points])


# plot_target([sample_points])


def KL(samples, d=-1):
    n = args.n
    radius = 2
    std = args.std
    delta_theta = 4 * np.pi / n

    centers_x = []
    centers_y = []
    centers_z = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                centers_x.append(radius * (i - n / 2) * delta_theta)
                centers_y.append(radius * (j - n / 2) * delta_theta)
                centers_z.append(radius * (k - n / 2) * delta_theta)

    centers_x = np.expand_dims(np.array(centers_x), 1)
    centers_y = np.expand_dims(np.array(centers_y), 1)
    centers_z = np.expand_dims(np.array(centers_z), 1)
    centers = np.concatenate([centers_x, centers_y, centers_z], 1)

    # s = np.array(samples)
    s = samples
    samplesP = np.zeros(s.shape[0])
    for i in range(n):
        samplesP = samplesP + scipy.stats.multivariate_normal.pdf(samples, centers[i],
                                                                  [[std, 0, 0], [0, std, 0], [0, 0, std]]) / 8

    samplesP = samplesP / sum(samplesP) + 1e-20
    pd = np.log(samplesP / (1 / float(s.shape[0]))) * samplesP

    KL = np.sum(pd)
    return KL


def JS(samples, d=-1):
    n = args.n
    radius = 2
    std = args.std
    delta_theta = 4 * np.pi / n
    centers_x = []
    centers_y = []
    centers_z = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                centers_x.append(radius * (i - n / 2) * delta_theta)
                centers_y.append(radius * (j - n / 2) * delta_theta)
                centers_z.append(radius * (k - n / 2) * delta_theta)

    centers_x = np.expand_dims(np.array(centers_x), 1)
    centers_y = np.expand_dims(np.array(centers_y), 1)
    centers_z = np.expand_dims(np.array(centers_z), 1)
    centers = np.concatenate([centers_x, centers_y, centers_z], 1)
    p = [1. / n for _ in range(n)]
    # s = np.array(samples)
    s = samples
    samplesP = np.zeros(s.shape[0])
    for i in range(n):
        samplesP = samplesP + scipy.stats.multivariate_normal.pdf(samples, centers[i],
                                                                  [[std, 0, 0], [0, std, 0], [0, 0, std]]) / 8
    samplesP = samplesP / sum(samplesP) + 1e-20
    print(sum(samplesP))

    JS = np.sum(
        np.log((1 / float(s.shape[0])) / ((samplesP + 1 / float(s.shape[0])) / 2)) / float(s.shape[0])) + np.sum(
        np.log(samplesP / ((samplesP + 1 / float(s.shape[0])) / 2)) * samplesP)
    return JS / 2


# Model params (most of hyper-params follow the original paper: https://arxiv.org/abs/1611.02163)
z_dim = 512
g_inp = z_dim
g_hid = 256
g_out = dset.size

d_inp = g_out
d_hid = 256
d_out = 1

lamda = 3e-2
minibatch_size = 512

unrolled_steps = args.us
d_learning_rate = args.dlr
z_learning_rate = args.zlr
g_learning_rate = args.glr
optim_betas = (0.9, 0.999)
num_iterations = args.num_iter
log_interval = args.log_iter
d_steps = 1
g_steps = 1
z_loop = 10
y_loop = 5
z_L2 = 0.0001
y_L2 = 0.0001
y_ln = 0.2
c = 0.005
log_time0=time.strftime("%Y_%m_%d_%H_%M_%S")
# os.mkdir(r'Cube_results/{}'.format(log_time0))
prefix = r"W{}_unrolled_steps-{}_{}_zloop{}_yloop{}_zL2{}_y_L2{}_yln{}_ylr{}_xlr{}_zlr{}_-prior_std-{:.2f}".format(
    WGAN, unrolled_steps, args.method,
    z_loop, y_loop,
    z_L2, y_L2, y_ln,
    d_learning_rate,
    g_learning_rate,
    z_learning_rate,
    np.std(dset.p))
print("Save file with prefix", prefix)


def noise_sampler(N, z_dim):
    return np.random.normal(size=[N, z_dim]).astype('float32')


###### MODELS: Generator model and discriminator model
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        # self.bn = nn.BatchNorm1d(hidden_size)

        if args.act == 'tanh':
            self.activation_fn = torch.tanh
        else:
            self.activation_fn = F.leaky_relu

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return self.map3(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = F.leaky_relu
        # self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        if args.method == 'WGAN':
            return self.map3(x)
        else:
            return torch.sigmoid(self.map3(x))

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()



def copy_parameter(y, z,time=0):
    # print(loss_L1(y.parameters()))
    # print(loss_L1(z.parameters()))
    if time ==0:
        for p, q in zip(y.parameters(), z.parameters()):
            p.data = q.clone().detach().requires_grad_()
    else:
        for p, q in zip(y.parameters(), z.parameters(time=time)):
            p.data = q.clone().detach().requires_grad_()
    # print(loss_L1(y.parameters()))
    # print('-'*80)
    return y


def d_loop():
    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(minibatch_size))

    if cuda:
        d_real_data = d_real_data.to(device)

    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    if cuda:
        target = target.to(device)
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    d_gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        d_gen_input = d_gen_input.to(device)

    with torch.no_grad():
        d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    if cuda:
        target = target.to(device)
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    d_grad = torch.autograd.grad(d_loss, D.parameters(), retain_graph=True)
    d_loss.backward()
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
    return d_real_error.item(), d_fake_error.item(), d_grad


def d_loop_LC(alphaF, alphaR, d_gen_input=None,only_loss=False):
    # 1. Train D on real+fake
    d_optimizer.zero_grad()
    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(minibatch_size))

    d_real_data = d_real_data.to(device)
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    target = target.to(device)
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    if d_gen_input is None:
        d_gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))

    d_gen_input = d_gen_input.to(device)
    if not only_loss:
        with torch.no_grad():
            d_fake_data = G(d_gen_input)
    else:
        d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    target = target.to(device)
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake
    lc_loss = torch.norm(d_real_decision - alphaF) ** 2 + torch.norm(d_fake_decision - alphaR) ** 2

    d_loss = d_real_error + d_fake_error + lamda * lc_loss
    if only_loss:
        return d_loss
    d_loss.backward()
    d_optimizer.step()
    alphaF = Exponential_moving_average(alphaF, d_real_decision).detach_()
    alphaR = Exponential_moving_average(alphaR, d_fake_decision).detach_()
    # Only optimizes D's parameters; changes based on stored gradients from backward()
    return d_real_error.item(), d_fake_error.item(), alphaF, alphaR


def d_loop_prox(D0, lamda=0.01, d_gen_input=None, only_loss=False):
    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(minibatch_size))
    d_real_data = d_real_data.to(device)
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    target = target.to(device)
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    if d_gen_input is None:
        d_gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
        d_gen_input = d_gen_input.to(device)
    if not only_loss:
        with torch.no_grad():
            d_fake_data = G(d_gen_input)
    else:
        d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    target = target.to(device)
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error + lamda * loss_2L2(D.parameters(), D0.parameters())
    if only_loss:
        return d_loss
    d_grad = torch.autograd.grad(d_loss, D.parameters(), retain_graph=True)

    d_loss.backward()
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
    return d_real_error.item(), d_fake_error.item(), d_grad


def d_unrolled_loop(d_gen_input=None):
    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(minibatch_size))
    if cuda:
        d_real_data = d_real_data.to(device)
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    if cuda:
        target = target.to(device)
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    if d_gen_input is None:
        d_gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        d_gen_input = d_gen_input.to(device)

    with torch.no_grad():
        d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    if cuda:
        target = target.to(device)
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    d_grad = torch.autograd.grad(d_loss, D.parameters(), retain_graph=True)
    d_loss.backward()
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
    return d_real_error.item(), d_fake_error.item(), d_grad


def d_unrolled_loop2(d_gen_input=None, only_loss=False):
    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(minibatch_size))
    d_real_data = d_real_data.to(device)
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    target = target.to(device)
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    if d_gen_input is None:
        d_gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
        d_gen_input = d_gen_input.to(device)
    if not only_loss:
        with torch.no_grad():
            d_fake_data = G(d_gen_input)
    else:
        d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    target = target.to(device)
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    if only_loss:
        return d_loss
    d_loss.backward()
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
    return d_real_error.item(), d_fake_error.item()


def d_unrolled_loop_W(d_gen_input=None):
    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        gen_input = gen_input.to(device)

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(minibatch_size))
    if cuda:
        d_real_data = d_real_data.to(device)

    #  1B: Train D on fake
    d_gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        d_gen_input = d_gen_input.to(device)

    with torch.no_grad():
        d_fake_data = G(d_gen_input)

    d_optimizer.zero_grad()
    d_fake_decision = D(d_fake_data)

    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    if cuda:
        target = target.to(device)
    # ones = true

    d_loss = torch.mean(d_fake_decision) - torch.mean(d_real_decision)
    d_loss.backward()
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
    for p in D.parameters():
        p.requires_grad_(False)
        p.clamp_(-c, c)
        p.requires_grad_(True)
    return torch.mean(d_real_decision).item(), torch.mean(d_fake_decision).item()

def g_loop_prox():
    # 2. Train G on D's response (but DO NOT train D on these labels)
    g_optimizer.zero_grad()

    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    gen_input = gen_input.to(device)

    if args.GN:
        backup = copy.deepcopy(D)
        for i in range(unrolled_steps):
            with torch.no_grad():
                D0 = copy.deepcopy(D)

            d_loop_prox(D0)

    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data)
    target = torch.ones_like(dg_fake_decision)
    target = target.to(device)
    g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
    if args.GN:
        d_loss = d_loop_prox(d_gen_input=gen_input, D0=copy.deepcopy(D), only_loss=True)
        dFy = torch.autograd.grad(g_error, D.parameters(), retain_graph=True)
        dfy = torch.autograd.grad(d_loss, D.parameters(), retain_graph=True)

        # 计算GN loss
        gFyfy = 0
        gfyfy = 0
        for Fy, fy in zip(dFy, dfy):
            gFyfy = gFyfy + torch.sum(Fy * fy)
            gfyfy = gfyfy + torch.sum(fy * fy) + 1e-10
        GN_loss = -gFyfy.detach() / gfyfy.detach() * d_loss
        GN_loss.backward()

    g_grad = torch.autograd.grad(g_error, G.parameters(), retain_graph=True)
    g_error.backward()
    g_optimizer.step()  # Only optimizes G's parameters

    if args.GN:
        D.load(backup)
        del backup
    return g_error.item(), g_grad


def g_loop_GAN():
    # 2. Train G on D's response (but DO NOT train D on these labels)
    g_optimizer.zero_grad()

    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    gen_input = gen_input.to(device)
    if args.GN:
        backup = copy.deepcopy(D)
        for i in range(unrolled_steps):
            d_unrolled_loop2(d_gen_input=gen_input)

    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data)
    target = torch.ones_like(dg_fake_decision)
    target = target.to(device)
    g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
    if args.GN:
        d_loss = d_unrolled_loop2(d_gen_input=gen_input, only_loss=True)
        dFy = torch.autograd.grad(g_error, D.parameters(), retain_graph=True)
        dfy = torch.autograd.grad(d_loss, D.parameters(), retain_graph=True)

        # 计算GN loss
        gFyfy = 0
        gfyfy = 0
        for Fy, fy in zip(dFy, dfy):
            gFyfy = gFyfy + torch.sum(Fy * fy)
            gfyfy = gfyfy + torch.sum(fy * fy) + 1e-10
        GN_loss = -gFyfy.detach() / gfyfy.detach() * d_loss
        GN_loss.backward()


    g_grad = torch.autograd.grad(g_error, G.parameters(), retain_graph=True)
    g_error.backward()
    g_optimizer.step()  # Only optimizes G's parameters
    if args.GN:
        D.load(backup)
        del backup
    return g_error.item(), g_grad


def g_loop_LC():
    # 2. Train G on D's response (but DO NOT train D on these labels)
    g_optimizer.zero_grad()

    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    gen_input = gen_input.to(device)
    if args.GN:
        backup = copy.deepcopy(D)
        alphaF0 = copy.deepcopy(alphaF)
        alphaR0 = copy.deepcopy(alphaR)
        for i in range(unrolled_steps):
            alphaF, alphaR = d_loop_LC(alphaF, alphaR, gen_input)

    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data)
    target = torch.ones_like(dg_fake_decision)
    target = target.to(device)
    g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
    d_loss = d_loop_LC(alphaF, alphaR, gen_input,only_loss=True)
    dFy = torch.autograd.grad(g_error, D.parameters(), retain_graph=True)
    dfy = torch.autograd.grad(d_loss, D.parameters(), retain_graph=True)

    # 计算GN loss
    gFyfy = 0
    gfyfy = 0
    for Fy, fy in zip(dFy, dfy):
        gFyfy = gFyfy + torch.sum(Fy * fy)
        gfyfy = gfyfy + torch.sum(fy * fy) + 1e-10
    GN_loss = -gFyfy.detach() / gfyfy.detach() * d_loss
    GN_loss.backward()




    g_grad = torch.autograd.grad(g_error, G.parameters(), retain_graph=True)
    g_error.backward()
    g_optimizer.step()  # Only optimizes G's parameters
    if args.GN:
        D.load(backup)
        alphaF = copy.deepcopy(alphaF0)
        alphaR = copy.deepcopy(alphaR0)
        del backup
    return g_error.item(), g_grad


def g_loop():
    # 2. Train G on D's response (but DO NOT train D on these labels)
    g_optimizer.zero_grad()

    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    gen_input = gen_input.to(device)

    if unrolled_steps > 0:
        # backup = copy.deepcopy(D)
        if rhg == 0:
            for i in range(unrolled_steps):
                d_loss = d_unrolled_loop(d_gen_input=gen_input)
        else:
            for i in range(unrolled_steps):
                d_loss = d_unrolled_loop2(d_gen_input=gen_input)

    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data)
    target = torch.ones_like(dg_fake_decision)
    if cuda:
        target = target.to(device)
    g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
    g_grad = torch.autograd.grad(g_error, G.parameters(), retain_graph=True)
    g_error.backward()
    g_optimizer.step()  # Only optimizes G's parameters

    # if unrolled_steps > 0:
    #     D.load(backup)
    #     del backup
    return g_error.item(), d_loss, g_grad


def g_loop_RHG():
    # 2. Train G on D's response (but DO NOT train D on these labels)
    for xp in G.parameters():
        if xp.grad is not None:
            xp.grad = None
    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        gen_input = gen_input.to(device)

    if unrolled_steps > 0:
        backup = copy.deepcopy(D)
    #  1A: Train D on real
    # Only optimizes D's parameters; changes based on stored gradients from backward()

    with higher.innerloop_ctx(D, d_optimizer, copy_initial_weights=True) as (fmodel, diffopt):
        for y_idx in range(unrolled_steps):
            d_real_data = torch.from_numpy(dset.sample(minibatch_size)).to(device)
            d_real_decision = fmodel(d_real_data)
            target = torch.ones_like(d_real_decision).to(device)
            d_real_error = criterion(d_real_decision, target)  # ones = true

            #  1B: Train D on fake
            d_gen_input = gen_input.to(device)
            d_fake_data = G(d_gen_input)
            d_fake_decision = fmodel(d_fake_data)
            target = torch.zeros_like(d_fake_decision).to(device)
            d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

            d_loss = d_real_error + d_fake_error
            diffopt.step(d_loss)

        g_fake_data = G(gen_input)
        dg_fake_decision = fmodel(g_fake_data)
        target = torch.ones_like(dg_fake_decision).to(device)
        g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
        grad_x = torch.autograd.grad(g_error, G.parameters(), retain_graph=True)
        for p, xp in zip(grad_x, G.parameters()):
            if xp.grad == None:
                xp.grad = p
            else:
                xp.grad += p
        g_optimizer.step()
        # copy_parameter(D, fmodel)
        # Only optimizes G's parameters

    # if unrolled_steps > 0:
    #     D.load(backup)
    #     del backup
    return g_error.item(), d_loss.item(), grad_x


def g_loop_IAPTT():
    # 2. Train G on D's response (but DO NOT train D on these labels)
    for xp in G.parameters():
        if xp.grad is not None:
            xp.grad = None
    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        gen_input = gen_input.to(device)

    if unrolled_steps > 0:
        backup = copy.deepcopy(D)
    #  1A: Train D on real
    # Only optimizes D's parameters; changes based on stored gradients from backward()

    with higher.innerloop_ctx(D, d_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
        F_k_bar_list = []
        for y_idx in range(unrolled_steps):
            d_real_data = torch.from_numpy(dset.sample(minibatch_size)).to(device)
            d_real_decision = fmodel(d_real_data)
            target = torch.ones_like(d_real_decision).to(device)
            d_real_error = criterion(d_real_decision, target)  # ones = true

            #  1B: Train D on fake
            d_gen_input = gen_input.to(device)
            d_fake_data = G(d_gen_input)
            d_fake_decision = fmodel(d_fake_data)
            target = torch.zeros_like(d_fake_decision).to(device)
            d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

            d_loss = d_real_error + d_fake_error
            diffopt.step(d_loss)
            g_fake_data = G(gen_input)
            dg_fake_decision = fmodel(g_fake_data)
            target = torch.ones_like(dg_fake_decision).to(device)
            g_error = criterion(dg_fake_decision, target)
            F_k_bar_list.append(g_error)

        p_max = F_k_bar_list.index(max(F_k_bar_list))

        g_fake_data = G(gen_input)
        dg_fake_decision = fmodel(g_fake_data,params=fmodel.parameters(time=p_max + 1))
        target = torch.ones_like(dg_fake_decision).to(device)
        g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
        grad_x = torch.autograd.grad(g_error, G.parameters(), retain_graph=True)
        grad_ia = torch.autograd.grad(g_error,fmodel.parameters(time=0), retain_graph=True)
        for p, xp in zip(grad_x, G.parameters()):
            if xp.grad == None:
                xp.grad = p
            else:
                xp.grad += p
        for p, xp in zip(grad_ia, D.parameters()):
            if xp.grad == None:
                xp.grad = p
            else:
                xp.grad += p
        g_optimizer.step()
        d_ia_optimizer.step()
        # copy_parameter(D, fmodel)
        # Only optimizes G's parameters

    # if unrolled_steps > 0:
    #     D.load(backup)
    #     del backup
    return g_error.item(), d_loss.item(), grad_x,p_max

def g_loop_GN():
    g_optimizer.zero_grad()

    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        gen_input = gen_input.to(device)

    if unrolled_steps > 0:
        backup = copy.deepcopy(D)
        if WGAN:
            for i in range(unrolled_steps):
                d_unrolled_loop_W(d_gen_input=gen_input)
        else:
            for i in range(unrolled_steps):
                d_unrolled_loop(d_gen_input=gen_input)

    # g_fake_data = G(gen_input)
    # dg_fake_decision = D(g_fake_data)
    # target = torch.ones_like(dg_fake_decision)
    # if cuda:
    #     target = target.to(device)
    # g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine

    # 计算D-real loss
    d_real_data = torch.from_numpy(dset.sample(minibatch_size))
    if cuda:
        d_real_data = d_real_data.to(device)
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    if cuda:
        target = target.to(device)
    d_real_error = criterion(d_real_decision, target)  # ones = true

    # 计算D-fake loss
    d_gen_input = gen_input
    if cuda:
        d_gen_input = d_gen_input.to(device)

    # with torch.no_grad():
    d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    if cuda:
        target = target.to(device)
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    # 计算f关于下层变量D的梯度
    dfy = torch.autograd.grad(d_loss, D.parameters(), retain_graph=True)

    # 计算G loss
    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data)
    target = torch.ones_like(dg_fake_decision)
    if cuda:
        target = target.to(device)
    g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
    # 计算F关于下层变量D的梯度
    dFy = torch.autograd.grad(g_error, D.parameters(), retain_graph=True)

    # 计算GN loss
    gFyfy = 0
    gfyfy = 0
    for Fy, fy in zip(dFy, dfy):
        gFyfy = gFyfy + torch.sum(Fy * fy)
        gfyfy = gfyfy + torch.sum(fy * fy) + 1e-10
    GN_loss = -gFyfy.detach() / gfyfy.detach() * d_loss

    g_grad = torch.autograd.grad(g_error + GN_loss, G.parameters(), retain_graph=True)
    g_grad_GN = torch.autograd.grad(GN_loss, G.parameters(), retain_graph=True)

    GN_loss.backward()
    g_error.backward()
    g_optimizer.step()  # Only optimizes G's parameters

    if unrolled_steps > 0:
        D.load(backup)
        del backup
    return g_error.item(), d_loss.item(), g_grad, g_grad_GN




def loop_RHG(G, D):
    total_time = 0
    total_hyper_time = 0
    val_losses = []

    def val_loss(params, hparams):

        g_fake_data = G(d_gen_input)
        if params == -1:
            dg_fake_decision = fmodel(g_fake_data)
        else:
            dg_fake_decision = fmodel(g_fake_data, params=params)
        target_real = torch.ones_like(dg_fake_decision).to(device)
        g_error = criterion(dg_fake_decision, target_real)
        val_losses.append(tonp(g_error))
        return g_error

    inner_losses = []

    def train_loss(params, hparams):

        d_fake_data = G(d_gen_input)
        if params == -1:
            d_real_decision = fmodel(d_real_data)
            d_fake_decision = fmodel(d_fake_data)
        else:
            d_real_decision = fmodel(d_real_data, params=params)
            d_fake_decision = fmodel(d_fake_data, params=params)
        target_real = torch.ones_like(d_real_decision).to(device)
        target_fake = torch.zeros_like(d_real_decision).to(device)

        d_real_error = criterion(d_real_decision, target_real)  # ones = true
        d_fake_error = criterion(d_fake_decision, target_fake)  # zeros = fake

        d_loss = d_real_error + d_fake_error

        return d_loss

    def inner_loop(hparams, params, optim, n_steps, log_interval, create_graph=False):
        params_history = [optim.get_opt_params(params)]

        for t in range(n_steps):
            params_history.append(optim(params_history[-1], hparams, create_graph=True))

            # if log_interval and (t % log_interval == 0 or t == n_steps-1):
            #     print('t={}, Loss: {:.6f}'.format(t, optim.curr_loss.item()))

        return params_history

    # 2. Train G on D's response (but DO NOT train D on these labels)

    # for xp in G.parameters():
    #     if xp.grad is not None:
    #         xp.grad = None
    d_real_data = torch.from_numpy(dset.sample(minibatch_size)).to(device)

    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    d_gen_input = gen_input.to(device)
    #
    # if unrolled_steps > 0:
    #     backup = copy.deepcopy(D)
    # #  1A: Train D on real
    # Only optimizes D's parameters; changes based on stored gradients from backward()

    g_optimizer.zero_grad()
    t0=time.time()

    fmodel = higher.monkeypatch(D, copy_initial_weights=True)#torch.device("cuda")
    inner_opt = hg.GradientDescent(train_loss, step_size=args.dlr)
    last_param = inner_loop(G.parameters(), D.parameters(), inner_opt, y_loop, log_interval=10)
    hyper_time = time.time()
    # print(torch.cuda.memory_allocated())

    if args.method == 'CG':
        # This is the approximation used in the paper CG stands for conjugate gradient
        cg_fp_map = hg.GradientDescent(loss_f=train_loss, step_size=args.dlr)
        hg.CG(last_param[-1], list(G.parameters()), K=5, fp_map=cg_fp_map, outer_loss=val_loss)
    elif args.method == 'Neumann':
        hg.fixed_point(last_param[-1], list(G.parameters()), K=5, fp_map=inner_opt,
                       outer_loss=val_loss)
    elif args.method == 'RHG':
        hg.reverse(last_param[-y_loop - 1:], list(G.parameters()), [inner_opt] * y_loop, val_loss)
    elif args.method == 'TRHG':
        hg.reverse(last_param[-int(y_loop / 2) - 1:], list(G.parameters()), [inner_opt] * int(y_loop / 2), val_loss)

    else:
        print('NO hypergradient!')

    g_optimizer.step()
    total_hyper_time += time.time() - hyper_time
    total_time += time.time() - t0
    copy_parameter_from_list(D, last_param[-1])
    # summary(G.to(device), input_size=(1,512), batch_size=256)
    # summary(D.to(device), input_size=(1,3), batch_size=256)

    g_error = val_loss(-1, -1)
    d_loss = train_loss(-1, -1)
    return g_error.item(), d_loss.item(), total_hyper_time, total_time



def loop_LC_RHG(G, D):
    total_time = 0
    total_hyper_time = 0
    val_losses = []

    def val_loss_lc(params, hparams):

        g_fake_data = G(d_gen_input)
        if params == -1:
            dg_fake_decision = fmodel(g_fake_data)
        else:
            dg_fake_decision = fmodel(g_fake_data, params=params)
        target_real = torch.ones_like(dg_fake_decision).to(device)
        g_error = criterion(dg_fake_decision, target_real)
        val_losses.append(tonp(g_error))
        return g_error

    inner_losses = []

    def train_loss_lc(params, hparams,alphaF=args.alphaF,alphaR=args.alphaR):

        d_fake_data = G(d_gen_input)
        if params == -1:
            d_real_decision = fmodel(d_real_data)
            d_fake_decision = fmodel(d_fake_data)
        else:
            d_real_decision = fmodel(d_real_data, params=params)
            d_fake_decision = fmodel(d_fake_data, params=params)
        target_real = torch.ones_like(d_real_decision).to(device)
        target_fake = torch.zeros_like(d_real_decision).to(device)

        d_real_error = criterion(d_real_decision, target_real)  # ones = true
        d_fake_error = criterion(d_fake_decision, target_fake)  # zeros = fake
        lc_loss = torch.norm(d_real_decision - alphaF) ** 2 + torch.norm(d_fake_decision - alphaR) ** 2
        d_loss = d_real_error + d_fake_error+lamda * lc_loss
        args.alphaF = Exponential_moving_average(alphaF, d_real_decision).detach_()
        args.alphaR = Exponential_moving_average(alphaR, d_fake_decision).detach_()
        return d_loss

    def inner_loop_lc(hparams, params, optim, n_steps, log_interval, create_graph=False):
        params_history = [optim.get_opt_params(params)]

        for t in range(n_steps):
            params_history.append(optim(params_history[-1], hparams, create_graph=True))

            # if log_interval and (t % log_interval == 0 or t == n_steps-1):
            #     print('t={}, Loss: {:.6f}'.format(t, optim.curr_loss.item()))

        return params_history

    # 2. Train G on D's response (but DO NOT train D on these labels)

    # for xp in G.parameters():
    #     if xp.grad is not None:
    #         xp.grad = None
    d_real_data = torch.from_numpy(dset.sample(minibatch_size)).to(device)

    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    d_gen_input = gen_input.to(device)
    #
    # if unrolled_steps > 0:
    #     backup = copy.deepcopy(D)
    # #  1A: Train D on real
    # Only optimizes D's parameters; changes based on stored gradients from backward()

    g_optimizer.zero_grad()
    t0=time.time()

    fmodel = higher.monkeypatch(D, copy_initial_weights=True)#, torch.device("cuda")
    inner_opt = hg.GradientDescent(train_loss_lc, step_size=args.dlr)
    last_param = inner_loop_lc(G.parameters(), D.parameters(), inner_opt, y_loop, log_interval=10)
    hyper_time = time.time()
    # print(torch.cuda.memory_allocated())

    if 'cg' in args.method:
        # This is the approximation used in the paper CG stands for conjugate gradient
        cg_fp_map = hg.GradientDescent(loss_f=train_loss_lc, step_size=args.dlr)
        hg.CG(last_param[-1], list(G.parameters()), K=5, fp_map=cg_fp_map, outer_loss=val_loss_lc)
    elif 'neumann' in args.method:
        hg.fixed_point(last_param[-1], list(G.parameters()), K=5, fp_map=inner_opt,
                       outer_loss=val_loss_lc)
    elif 'rhg' in args.method:
        hg.reverse(last_param[-y_loop - 1:], list(G.parameters()), [inner_opt] * y_loop, val_loss_lc)
    elif 'trhg' in args.method:
        hg.reverse(last_param[-int(y_loop / 2) - 1:], list(G.parameters()), [inner_opt] * int(y_loop / 2), val_loss_lc)

    else:
        print('NO hypergradient!')

    g_optimizer.step()
    total_hyper_time += time.time() - hyper_time
    total_time += time.time() - t0
    copy_parameter_from_list(D, last_param[-1])
    # summary(G.to(device), input_size=(1,512), batch_size=256)
    # summary(D.to(device), input_size=(1,3), batch_size=256)

    g_error = val_loss_lc(-1, -1)
    d_loss = train_loss_lc(-1, -1)
    return g_error.item(), d_loss.item(), total_hyper_time, total_time



def g_loop_GN_prox():
    g_optimizer.zero_grad()

    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        gen_input = gen_input.to(device)

    if unrolled_steps > 0:
        backup = copy.deepcopy(D)
        for i in range(unrolled_steps):
            d_loop_prox(d_gen_input=gen_input, D0=copy.deepcopy(D))

    # g_fake_data = G(gen_input)
    # dg_fake_decision = D(g_fake_data)
    # target = torch.ones_like(dg_fake_decision)
    # if cuda:
    #     target = target.to(device)
    # g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine

    # 计算D-real loss
    d_real_data = torch.from_numpy(dset.sample(minibatch_size))
    if cuda:
        d_real_data = d_real_data.to(device)
    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    if cuda:
        target = target.to(device)
    d_real_error = criterion(d_real_decision, target)  # ones = true

    # 计算D-fake loss
    d_gen_input = gen_input
    if cuda:
        d_gen_input = d_gen_input.to(device)

    # with torch.no_grad():
    d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    if cuda:
        target = target.to(device)
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error + 0.01 * torch.sqrt(loss_2L2(D.parameters(), backup.parameters()))
    # 计算f关于下层变量D的梯度
    dfy = torch.autograd.grad(d_loss, D.parameters(), retain_graph=True)

    # 计算G loss
    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data)
    target = torch.ones_like(dg_fake_decision)
    if cuda:
        target = target.to(device)
    g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
    # 计算F关于下层变量D的梯度
    dFy = torch.autograd.grad(g_error, D.parameters(), retain_graph=True)

    # 计算GN loss
    gFyfy = 0
    gfyfy = 0
    for Fy, fy in zip(dFy, dfy):
        gFyfy = gFyfy + torch.sum(Fy * fy)
        gfyfy = gfyfy + torch.sum(fy * fy)
    GN_loss = -gFyfy.detach() / gfyfy.detach() * d_loss

    g_grad = torch.autograd.grad(g_error + GN_loss, G.parameters(), retain_graph=True)
    g_grad_GN = torch.autograd.grad(GN_loss, G.parameters(), retain_graph=True)

    GN_loss.backward()
    g_error.backward()
    g_optimizer.step()  # Only optimizes G's parameters

    if unrolled_steps > 0:
        D.load(backup)
        del backup
    return g_error.item(), d_loss.item(), g_grad, g_grad_GN


def WGAN_loop():
    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    gen_input = gen_input.to(device)
    # 1. Train D on real+fake
    if args.GN:
        for i in range(1):
            d_optimizer.zero_grad()
            #  1A: Train D on real
            d_real_data = torch.from_numpy(dset.sample(minibatch_size))
            d_real_data = d_real_data.to(device)

            #  1B: Train D on fake
            d_gen_input = gen_input
            with torch.no_grad():
                d_fake_data = G(d_gen_input)

            d_fake_decision = D(d_fake_data)
            d_real_decision = D(d_real_data)
            # target = torch.ones_like(d_real_decision)
            # target = target.to(device)
            # ones = true

            d_loss = torch.mean(d_fake_decision) - torch.mean(d_real_decision)
            if i == 0:
                d_grad = torch.autograd.grad(d_loss, D.parameters(), retain_graph=True)
            d_loss.backward()
            d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
            for p in D.parameters():
                p.requires_grad_(False)
                p.clamp_(-c, c)
                p.requires_grad_(True)


    else:
        for i in range(unrolled_steps):
            d_optimizer.zero_grad()
            #  1A: Train D on real
            d_real_data = torch.from_numpy(dset.sample(minibatch_size))
            d_real_data = d_real_data.to(device)

            #  1B: Train D on fake
            d_gen_input = gen_input
            with torch.no_grad():
                d_fake_data = G(d_gen_input)

            d_optimizer.zero_grad()
            d_fake_decision = D(d_fake_data)

            d_real_decision = D(d_real_data)
            # target = torch.ones_like(d_real_decision)
            # target = target.to(device)
            # ones = true

            d_loss = torch.mean(d_fake_decision) - torch.mean(d_real_decision)
            if i == 0:
                d_grad = torch.autograd.grad(d_loss, D.parameters(), retain_graph=True)
            d_loss.backward()
            d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
            for p in D.parameters():
                p.requires_grad_(False)
                p.clamp_(-c, c)
                p.requires_grad_(True)
    p, r, F1 = D_acc(D, torch.from_numpy(dset.sample(minibatch_size)).to(device),
                     G(torch.from_numpy(noise_sampler(minibatch_size, g_inp)).to(device)))

    # 2. Train G on D's response (but DO NOT train D on these labels)

    # G loss

    # D_z loss

    if args.GN:
        backup = copy.deepcopy(D)
        for i in range(args.us):
            d_optimizer.zero_grad()
            #  1A: Train D on real
            d_real_data = torch.from_numpy(dset.sample(minibatch_size))
            d_real_data = d_real_data.to(device)

            #  1B: Train D on fake
            d_gen_input = gen_input
            with torch.no_grad():
                d_fake_data = G(d_gen_input)

            d_fake_decision = D(d_fake_data)
            d_real_decision = D(d_real_data)
            # target = torch.ones_like(d_real_decision)
            # target = target.to(device)
            # ones = true

            d_loss = torch.mean(d_fake_decision) - torch.mean(d_real_decision)
            if i == 0:
                d_grad = torch.autograd.grad(d_loss, D.parameters(), retain_graph=True)
            d_loss.backward()
            d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
            for p in D.parameters():
                p.requires_grad_(False)
                p.clamp_(-c, c)
                p.requires_grad_(True)

    g_optimizer.zero_grad()
    d_fake_data = G(d_gen_input)
    d_fake_decision = D(d_fake_data)

    # target = torch.zeros_like(d_fake_decision)
    # if cuda:
    #     target = target.to(device)
    # d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_real_decision = D(d_real_data)
    # target = torch.ones_like(d_real_decision)
    # if cuda:
    #     target = target.to(device)
    # d_real_error = criterion(d_real_decision, target)  # ones = true

    g_loss = -torch.mean(d_fake_decision)
    if args.GN:
        d_real_decision = D(d_real_data)
        d_loss = torch.mean(d_fake_decision) - torch.mean(d_real_decision)

        dFy = torch.autograd.grad(g_loss, D.parameters(), retain_graph=True)
        dfy = torch.autograd.grad(d_loss, D.parameters(), retain_graph=True)

        # 计算GN loss
        gFyfy = 0
        gfyfy = 0
        for Fy, fy in zip(dFy, dfy):
            gFyfy = gFyfy + torch.sum(Fy * fy)
            gfyfy = gfyfy + torch.sum(fy * fy) + 1e-10
        GN_loss = -gFyfy.detach() / gfyfy.detach() * d_loss
        GN_loss.backward(retain_graph=True)
        D.load(backup)
        del backup

    g_grad = torch.autograd.grad(g_loss, G.parameters(), retain_graph=True)
    g_loss.backward()
    g_optimizer.step()  # Only optimizes G's parameters
    # for p in G.parameters():
    #     p.requires_grad_(False)
    #     p.clamp_(-c,c)
    #     p.requires_grad_(True)

    return g_loss.item(), d_loss.item(), g_grad, d_grad, p, r, F1



def loop_prox_RHG(G, D, D0):
    total_time = 0
    total_hyper_time = 0
    val_losses = []

    def val_loss_prox(params, hparams):

        g_fake_data = G(d_gen_input)
        if params == -1:
            dg_fake_decision = fmodel(g_fake_data)
        else:
            dg_fake_decision = fmodel(g_fake_data, params=params)
        target_real = torch.ones_like(dg_fake_decision).to(device)
        # print(target_real.shape)
        if torch.max(dg_fake_decision).data>=1 or torch.min(dg_fake_decision).data<=0:
            g_error = criterion(torch.sigmoid(dg_fake_decision), target_real)
        else:
            # print(dg_fake_decision)
            g_error = criterion(dg_fake_decision, target_real)
        val_losses.append(g_error.item())
        return g_error

    inner_losses = []

    def train_loss_prox(params, hparams,lamda=0.001):

        d_fake_data = G(d_gen_input)
        if params == -1:
            d_real_decision = fmodel(d_real_data)
            d_fake_decision = fmodel(d_fake_data)
        else:
            d_real_decision = fmodel(d_real_data, params=params)
            d_fake_decision = fmodel(d_fake_data, params=params)
        target_real = torch.ones_like(d_real_decision).to(device)
        target_fake = torch.zeros_like(d_real_decision).to(device)

        d_real_error = criterion(d_real_decision, target_real)  # ones = true
        d_fake_error = criterion(d_fake_decision, target_fake)  # zeros = fake

        d_loss = d_real_error + d_fake_error+lamda * loss_2L2(fmodel.parameters() if params==-1 else params, D0.parameters())

        return d_loss

    def inner_loop_prox(hparams, params, optim, n_steps, log_interval, create_graph=False):
        params_history = [optim.get_opt_params(params)]

        for t in range(n_steps):
            params_history.append(optim(params_history[-1], hparams, create_graph=True))

            # if log_interval and (t % log_interval == 0 or t == n_steps-1):
            #     print('t={}, Loss: {:.6f}'.format(t, optim.curr_loss.item()))

        return params_history

    # 2. Train G on D's response (but DO NOT train D on these labels)

    # for xp in G.parameters():
    #     if xp.grad is not None:
    #         xp.grad = None
    d_real_data = torch.from_numpy(dset.sample(minibatch_size)).to(device)

    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    d_gen_input = gen_input.to(device)
    #
    # if unrolled_steps > 0:
    #     backup = copy.deepcopy(D)
    # #  1A: Train D on real
    # Only optimizes D's parameters; changes based on stored gradients from backward()

    g_optimizer.zero_grad()
    t0=time.time()

    fmodel = higher.monkeypatch(D, copy_initial_weights=True)#, torch.device("cuda")
    inner_opt = hg.GradientDescent(train_loss_prox, step_size=args.dlr)
    last_param = inner_loop_prox(G.parameters(), D.parameters(), inner_opt, y_loop, log_interval=10)
    hyper_time = time.time()
    # print(torch.cuda.memory_allocated())

    if 'cg' in args.method:
        # This is the approximation used in the paper CG stands for conjugate gradient
        cg_fp_map = hg.GradientDescent(loss_f=train_loss_prox, step_size=args.dlr)
        hg.CG(last_param[-1], list(G.parameters()), K=5, fp_map=cg_fp_map, outer_loss=val_loss_prox)
    elif 'neumann' in args.method:
        hg.fixed_point(last_param[-1], list(G.parameters()), K=5, fp_map=inner_opt,
                       outer_loss=val_loss_prox)
    elif 'rhg' in args.method:
        hg.reverse(last_param[-y_loop - 1:], list(G.parameters()), [inner_opt] * y_loop, val_loss_prox)
    elif 'trhg' in args.method:
        hg.reverse(last_param[-int(y_loop / 2) - 1:], list(G.parameters()), [inner_opt] * int(y_loop / 2), val_loss_prox)

    else:
        print('NO hypergradient!')
    # torch.nn.utils.clip_grad_norm_(G.parameters(),c)
    g_optimizer.step()
    total_hyper_time += time.time() - hyper_time
    total_time += time.time() - t0
    copy_parameter_from_list(D, last_param[-1])
    # summary(G.to(device), input_size=(1,512), batch_size=256)
    # summary(D.to(device), input_size=(1,3), batch_size=256)

    g_error = val_loss_prox(last_param[-1], -1)
    d_loss = train_loss_prox(last_param[-1], -1)
    return g_error.item(), d_loss.item(), total_hyper_time, total_time
    # return 0, 0, total_hyper_time, total_time






def WGAN_loop_GP_RHG(G,D):
    total_time = 0
    total_hyper_time = 0
    val_losses = []
    def val_loss_w(params, hparams):

        g_fake_data = G(gen_input)
        if params == -1:
            dg_fake_decision = fmodel(g_fake_data)
        else:
            dg_fake_decision = fmodel(g_fake_data, params=params)
        g_error = -torch.mean(dg_fake_decision)
        val_losses.append(tonp(g_error))
        return g_error


    def train_loss_w(params, hparams):
        d_real_data = torch.from_numpy(dset.sample(minibatch_size)).to(device)
        d_fake_data = G(gen_input)
        if params == -1:
            d_real_decision = fmodel(d_real_data)
            d_fake_decision = fmodel(d_fake_data)
        else:
            d_real_decision = fmodel(d_real_data, params=params)
            d_fake_decision = fmodel(d_fake_data, params=params)
        d_loss = torch.mean(d_fake_decision) - torch.mean(d_real_decision)
        if args.GP:
            gradt = torch.autograd.grad(torch.mean(d_fake_decision), d_fake_data, retain_graph=True)
            grad = 0
            for g in gradt:
                grad = grad + torch.norm(g - 1)
            GP_loss = args.lamda * (grad) ** 2
            d_loss = d_loss + GP_loss
        return d_loss

    def inner_loop_w(hparams, params, optim, n_steps, log_interval, create_graph=False):
        params_history = [optim.get_opt_params(params)]

        for t in range(n_steps):
            temp_param=optim(params_history[-1], hparams, create_graph=True)
            # if not args.GP:
            #     for p in temp_param:
            #         # p.requires_grad_(False)
            #         p.clamp_(-c, c)
            #         # p.requires_grad_(True)
            params_history.append(temp_param)

            # if log_interval and (t % log_interval == 0 or t == n_steps-1):
            #     print('t={}, Loss: {:.6f}'.format(t, optim.curr_loss.item()))

        return params_history


    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    gen_input = gen_input.to(device)
    g_optimizer.zero_grad()
    t0 = time.time()
    fmodel = higher.monkeypatch(D, copy_initial_weights=True)#, torch.device("cuda")
    inner_opt = hg.GradientDescent(train_loss_w, step_size=args.dlr)
    last_param = inner_loop_w(G.parameters(), D.parameters(), inner_opt, y_loop, log_interval=10)
    hyper_time = time.time()
    if 'cg' in args.method:
        # This is the approximation used in the paper CG stands for conjugate gradient
        cg_fp_map = hg.GradientDescent(loss_f=train_loss_w, step_size=args.dlr)
        hg.CG(last_param[-1], list(G.parameters()), K=5, fp_map=cg_fp_map, outer_loss=val_loss_w)
    elif 'neumann' in args.method:
        hg.fixed_point(last_param[-1], list(G.parameters()), K=5, fp_map=inner_opt,
                       outer_loss=val_loss_w)
    elif 'rhg' in args.method:
        hg.reverse(last_param[-y_loop - 1:], list(G.parameters()), [inner_opt] * y_loop, val_loss_w)
    elif 'trhg' in args.method:
        hg.reverse(last_param[-int(y_loop / 2) - 1:], list(G.parameters()), [inner_opt] * int(y_loop / 2), val_loss_w)

    else:
        print('NO hypergradient!')
    g_optimizer.step()
    total_hyper_time += time.time() - hyper_time
    total_time += time.time() - t0
    copy_parameter_from_list(D, last_param[-1])
    g_error = val_loss_w(-1, -1)
    d_loss = train_loss_w(-1, -1)
    p, r, F1 = D_acc(D, torch.from_numpy(dset.sample(minibatch_size)).to(device),
                     G(torch.from_numpy(noise_sampler(minibatch_size, g_inp)).to(device)))

    return g_error.item(), d_loss.item(), p, r, F1



def GN_loop():
    # 1. Train D on real+fake

    gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        gen_input = gen_input.to(device)

    #  1A: Train D on real
    d_real_data = torch.from_numpy(dset.sample(minibatch_size))
    if cuda:
        d_real_data = d_real_data.to(device)

    #  1B: Train D on fake
    d_gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
    if cuda:
        d_gen_input = d_gen_input.to(device)

    with torch.no_grad():
        d_fake_data = G(d_gen_input)
    for i in range(z_loop):
        d_optimizer.zero_grad()
        d_fake_decision = D(d_fake_data)
        target = torch.zeros_like(d_fake_decision)
        if cuda:
            target = target.to(device)
        d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

        d_real_decision = D(d_real_data)
        target = torch.ones_like(d_real_decision)
        if cuda:
            target = target.to(device)
        d_real_error = criterion(d_real_decision, target)  # ones = true

        d_loss = d_real_error + d_fake_error
        d_loss.backward()
        d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

    # 2. Train G on D's response (but DO NOT train D on these labels)
    g_optimizer.zero_grad()

    d_fake_decision = D(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    if cuda:
        target = target.to(device)
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_real_decision = D(d_real_data)
    target = torch.ones_like(d_real_decision)
    if cuda:
        target = target.to(device)
    d_real_error = criterion(d_real_decision, target)  # ones = true

    # G loss
    g_fake_data = G(gen_input)
    dg_fake_decision = D(g_fake_data)
    target = torch.ones_like(dg_fake_decision)
    if cuda:
        target = target.to(device)
    g_loss_ = criterion(dg_fake_decision, target)

    d_loss_ = d_real_error + d_fake_error

    dfy = torch.autograd.grad(d_loss_, D.parameters(), retain_graph=True)
    dFy = torch.autograd.grad(g_loss_, D.parameters(), retain_graph=True)
    gFyfy = 0
    gfyfy = 0
    for Fy, fy in zip(dFy, dfy):
        gFyfy = gFyfy + torch.sum(Fy * fy)
        gfyfy = gfyfy + torch.sum(fy * fy)

    g_loss = g_loss_

    g_optimizer.step()  # Only optimizes G's parameters

    return g_loss.item()


def g_sample():
    with torch.no_grad():
        gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
        if cuda:
            gen_input = gen_input.to(device)
        g_fake_data = G(gen_input)
        return g_fake_data.cpu().numpy()


# In[13]:


G = (Generator(input_size=g_inp, hidden_size=g_hid, output_size=g_out))
D = (Discriminator(input_size=d_inp, hidden_size=d_hid, output_size=d_out))
D_z = Discriminator(input_size=d_inp, hidden_size=d_hid, output_size=d_out)
if args.method=='SN':
    G=add_sn(G)
if cuda:
    G = G.to(device)
    D = D.to(device)
    D_z = D_z.to(device)

criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)

d_ia_optimizer = optim.Adam(D.parameters(), lr=z_learning_rate, betas=optim_betas) # IAPTT ,IA

d_z_optimizer = optim.Adam(D_z.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)
# d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate)
#
# d_ia_optimizer = optim.SGD(D.parameters(), lr=z_learning_rate) # IAPTT ,IA
#
# d_z_optimizer = optim.SGD(D_z.parameters(), lr=d_learning_rate)
# g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate)

samples = []
glist = []
glist_GN = []
glistl2 = []
glistl1 = []
dlistl2 = []
dlistl1 = []
dlist = []
KLlist = []
JSlist = []
gradlist = []
dgradlist = []
gradlistl1 = []
dgradlistl1 = []
timelist = []
fidlist = []
eachloopjslist = []
gmodel = [None] * len(G.state_dict())
gmodelist = []
ggradmodel = [None] * len(G.state_dict())
ggradmodelist = []
plist = []
rlist = []
F1list = []
for it in range(num_iterations):
    t0 = time.time()
    if args.method == 'BIM':
        regd = 1. ** (it + 1)
        # g_info = BIM_loop(regd)
    elif args.method == 'WGAN':
        g_info, d_loss, g_grad, d_grad, p, r, F1 = WGAN_loop()
        # glist.append(g_grad)
    elif args.method == 'WGAN_rhg' or args.method == 'WGAN_cg' or args.method == 'WGAN_neumann' or args.method == 'WGAN_bda':
        t0 = time.time()

        g_info, d_loss, p, r, F1 = WGAN_loop_GP_RHG(G,D)
        timelist.append(time.time() - t0)

        # glist.append(g_grad)

    elif args.method == 'GN':
        # print('GN')
        d_infos = [[0, 0], [0, 0]]
        for d_index in range(1):
            if WGAN:
                d_info_r, d_info_f, d_grad = d_unrolled_loop()
            else:
                d_info_r, d_info_f, d_grad = d_loop()
            d_info = [d_info_r, d_info_f]
            d_infos.append(d_info)
        d_infos = np.mean(d_infos, 0)
        d_real_loss, d_fake_loss = d_infos
        p, r, F1 = D_acc(D, torch.from_numpy(dset.sample(minibatch_size)).to(device),
                         G(torch.from_numpy(noise_sampler(minibatch_size, g_inp)).to(device)))

        g_infos = []
        for g_index in range(g_steps):
            g_info, d_loss, g_grad, g_grad_GN = g_loop_GN()
            g_infos.append(g_info)
        g_infos = np.mean(g_infos)
        g_loss = g_infos
        # gradlist.append(g_grad)

    elif args.method == 'prox':
        d_infos = [[0, 0], [0, 0]]
        if args.GN:
            for d_index in range(1):
                with torch.no_grad():
                    D0 = copy.deepcopy(D)
                d_info_r, d_info_f, d_grad = d_loop_prox(D0)
                d_info = [d_info_r, d_info_f]
                d_infos.append(d_info)
        else:
            for d_index in range(args.us):
                with torch.no_grad():
                    D0 = copy.deepcopy(D)
                d_info_r, d_info_f, d_grad = d_loop_prox(D0)
                d_info = [d_info_r, d_info_f]
                d_infos.append(d_info)
        d_infos = np.mean(d_infos, 0)
        d_real_loss, d_fake_loss = d_infos
        d_loss = d_real_loss + d_fake_loss
        p, r, F1 = D_acc(D, torch.from_numpy(dset.sample(minibatch_size)).to(device),
                         G(torch.from_numpy(noise_sampler(minibatch_size, g_inp)).to(device)))

        g_infos = []
        for g_index in range(g_steps):
            g_info, g_grad = g_loop_prox()
            g_infos.append(g_info)
        g_infos = np.mean(g_infos)
        g_loss = g_infos
        # gradlist.append(g_grad)
    elif args.method == 'prox_rhg' or args.method == 'prox_cg' or args.method == 'prox_neumann' or args.method == 'porx_bda':
        t0 = time.time()

        d_infos = [[0, 0], [0, 0]]
        with torch.no_grad():
            D0 = copy.deepcopy(D).to(device)
        g_info, d_loss, hyper_time_bi, time_bi = loop_prox_RHG(G, D,D0)
        p, r, F1 = D_acc(D, torch.from_numpy(dset.sample(minibatch_size)).to(device),
                         G(torch.from_numpy(noise_sampler(minibatch_size, g_inp)).to(device)))

        timelist.append(time.time() - t0)
        # gradlist.append(g_grad)
    elif args.method == 'GAN' or args.method =='SN':
        d_infos = [[0, 0], [0, 0]]
        if args.GN:
            for d_index in range(1):
                d_info_r, d_info_f, d_grad = d_loop()
                d_info = [d_info_r, d_info_f]
                d_infos.append(d_info)
                if d_index == 0:
                    d_grad_0 = d_grad
        else:
            for d_index in range(unrolled_steps):
                d_info_r, d_info_f, d_grad = d_loop()
                d_info = [d_info_r, d_info_f]
                d_infos.append(d_info)
                if d_index == 0:
                    d_grad_0 = d_grad
        d_grad = d_grad_0
        d_infos = np.mean(d_infos, 0)
        d_real_loss, d_fake_loss = d_infos

        g_infos = []
        p, r, F1 = D_acc(D, torch.from_numpy(dset.sample(minibatch_size)).to(device),
                         G(torch.from_numpy(noise_sampler(minibatch_size, g_inp)).to(device)))

        for g_index in range(g_steps):
            g_info, g_grad = g_loop_GAN()
            g_infos.append(g_info)
        g_infos = np.mean(g_infos)
        g_loss = g_infos
        d_loss = d_real_loss + d_fake_loss
        # gradlist.append(g_grad)
    elif args.method == 'LC_rhg' or args.method == 'LC_cg' or args.method == 'LC_neumann' or args.method == 'LC_bda':
        t0 = time.time()

        d_infos = [[0, 0], [0, 0]]
        with torch.no_grad():
            D0 = copy.deepcopy(D)
        g_info, d_loss, hyper_time_bi, time_bi = loop_LC_RHG(G, D)
        p, r, F1 = D_acc(D, torch.from_numpy(dset.sample(minibatch_size)).to(device),
                         G(torch.from_numpy(noise_sampler(minibatch_size, g_inp)).to(device)))

        timelist.append(time.time() - t0)

    elif args.method == 'LC':
        d_infos = [[0, 0], [0, 0]]
        if args.GN:
            for d_index in range(1):
                d_info1, d_info2, alphaF, alphaR = d_loop_LC(alphaF, alphaR)
                d_infos.append([d_info1, d_info2])
        else:
            for d_index in range(args.us):
                d_info1, d_info2, alphaF, alphaR = d_loop_LC(alphaF, alphaR)
                d_infos.append([d_info1, d_info2])
        d_infos = np.mean(d_infos, 0)
        d_real_loss, d_fake_loss = d_infos
        p, r, F1 = D_acc(D, torch.from_numpy(dset.sample(minibatch_size)).to(device),
                         G(torch.from_numpy(noise_sampler(minibatch_size, g_inp)).to(device)))
        g_infos = []
        for g_index in range(g_steps):
            g_info, g_grad = g_loop_LC()
            g_infos.append(g_info)
        g_infos = np.mean(g_infos)
        g_loss = g_infos
        d_loss = d_real_loss + d_fake_loss
        # gradlist.append(g_grad)
        # gradlist.append(g_grad)
    elif args.method == 'IAPTT':
    # 2. Train G on D's response (but DO NOT train D on these labels)
        #print('IAPTT excuted!')
        for xp in G.parameters():
            if xp.grad is not None:
                xp.grad = None
        for xp in D.parameters():
            if xp.grad is not None:
                xp.grad = None
        gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
        if cuda:
            gen_input = gen_input.to(device)

        #  1A: Train D on real
        # Only optimizes D's parameters; changes based on stored gradients from backward()

        with higher.innerloop_ctx(D, d_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
            F_k_bar_list = []
            d_infos = [[0, 0], [0, 0]]
            for y_idx in range(args.us):
                d_real_data = torch.from_numpy(dset.sample(minibatch_size)).to(device)
                d_real_decision = fmodel(d_real_data)
                target = torch.ones_like(d_real_decision).to(device)
                d_real_error = criterion(d_real_decision, target)  # ones = true

                #  1B: Train D on fake
                d_gen_input = gen_input.to(device)
                d_fake_data = G(d_gen_input)
                d_fake_decision = fmodel(d_fake_data)
                target = torch.zeros_like(d_fake_decision).to(device)
                d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

                d_loss = d_real_error + d_fake_error
                diffopt.step(d_loss)
                g_fake_data = G(gen_input)
                dg_fake_decision = fmodel(g_fake_data)
                target = torch.ones_like(dg_fake_decision).to(device)
                g_k_bar_error = criterion(dg_fake_decision, target)
                F_k_bar_list.append(g_k_bar_error)


            d_infos.append([d_real_error.item(), d_fake_error.item()])
            d_infos = np.mean(d_infos, 0)
            d_real_loss, d_fake_loss = d_infos
            p, r, F1 = D_acc(fmodel, torch.from_numpy(dset.sample(minibatch_size)).to(device),
                    G(torch.from_numpy(noise_sampler(minibatch_size, g_inp)).to(device)))

            #p_max = F_k_bar_list.index(max(F_k_bar_list))
            p_max = args.us-1
            #print(p_max)
            g_fake_data = G(gen_input)
            dg_fake_decision = fmodel(g_fake_data,params=fmodel.parameters(time=p_max+1))
            target = torch.ones_like(dg_fake_decision).to(device)
            g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
            grad_x = torch.autograd.grad(g_error, G.parameters(), retain_graph=True)
            grad_ia = torch.autograd.grad(g_error,fmodel.parameters(time=0), retain_graph=True)
            for px, xp in zip(grad_x, G.parameters()):
                if xp.grad == None:

                    xp.grad = px
                else:
                    xp.grad += px
                #print(px)
            for py, yp in zip(grad_ia, D.parameters()):
                if yp.grad == None:
                    yp.grad = py
                    #if it==0:
                    #print(py)
                else:
                    yp.grad += py
            g_grad = grad_x
            g_loss = g_error.item()
            d_loss = d_real_loss + d_fake_loss
            g_info = g_error.item()
            g_optimizer.step()
            #copy_parameter(D, fmodel)
            d_ia_optimizer.step()
            #d_ia_optimizer.step()

    elif args.method == 'IAPTTOS':
    # 2. Train G on D's response (but DO NOT train D on these labels)
        #print('IAPTT excuted!')
        for xp in G.parameters():
            if xp.grad is not None:
                xp.grad = None
        for xp in D.parameters():
            if xp.grad is not None:
                xp.grad = None
        gen_input = torch.from_numpy(noise_sampler(minibatch_size, g_inp))
        if cuda:
            gen_input = gen_input.to(device)

        #  1A: Train D on real
        # Only optimizes D's parameters; changes based on stored gradients from backward()

        with higher.innerloop_ctx(D, d_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
            F_k_bar_list = []
            d_infos = [[0, 0], [0, 0]]
            for y_idx in range(args.us):
                d_real_data = torch.from_numpy(dset.sample(minibatch_size)).to(device)
                d_real_decision = fmodel(d_real_data)
                target = torch.ones_like(d_real_decision).to(device)
                d_real_error = criterion(d_real_decision, target)  # ones = true

                #  1B: Train D on fake
                d_gen_input = gen_input.to(device)
                d_fake_data = G(d_gen_input)
                d_fake_decision = fmodel(d_fake_data)
                target = torch.zeros_like(d_fake_decision).to(device)
                d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

                d_loss = d_real_error + d_fake_error
                diffopt.step(d_loss)
                g_fake_data = G(gen_input)
                dg_fake_decision = fmodel(g_fake_data)
                target = torch.ones_like(dg_fake_decision).to(device)
                g_k_bar_error = criterion(dg_fake_decision, target)
                F_k_bar_list.append(g_k_bar_error)


            d_infos.append([d_real_error.item(), d_fake_error.item()])
            d_infos = np.mean(d_infos, 0)
            d_real_loss, d_fake_loss = d_infos
            p, r, F1 = D_acc(fmodel, torch.from_numpy(dset.sample(minibatch_size)).to(device),
                    G(torch.from_numpy(noise_sampler(minibatch_size, g_inp)).to(device)))

            p_max = F_k_bar_list.index(max(F_k_bar_list))
            #print(p_max)
            g_fake_data = G(gen_input)
            dg_fake_decision = fmodel(g_fake_data,params=fmodel.parameters(time=p_max+1))
            target = torch.ones_like(dg_fake_decision).to(device)
            g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
            grad_x = torch.autograd.grad(g_error, G.parameters(), retain_graph=True)
            grad_ia = torch.autograd.grad(g_error,fmodel.parameters(time=0), retain_graph=True)
            for px, xp in zip(grad_x, G.parameters()):
                if xp.grad == None:

                    xp.grad = px
                else:
                    xp.grad += px
                #print(px)
            for py, yp in zip(grad_ia, D.parameters()):
                if yp.grad == None:
                    yp.grad = py
                    #if it==0:
                    #print(py)
                else:
                    yp.grad += py
            g_grad = grad_x
            g_loss = g_error.item()
            d_loss = d_real_loss + d_fake_loss
            g_info = g_error.item()
            g_optimizer.step()
            #copy_parameter(D, fmodel)
            d_ia_optimizer.step()
        with higher.innerloop_ctx(D, d_optimizer) as (fmodel, diffopt):
            d_real_data = torch.from_numpy(dset.sample(minibatch_size)).to(device)
            d_real_decision = fmodel(d_real_data)
            target = torch.ones_like(d_real_decision).to(device)
            d_real_error_back = criterion(d_real_decision, target)  # ones = true

            #  1B: Train D on fake
            d_gen_input = gen_input.to(device)
            d_fake_data = G(d_gen_input)
            d_fake_decision = fmodel(d_fake_data)
            target = torch.zeros_like(d_fake_decision).to(device)
            d_fake_error_back = criterion(d_fake_decision, target)  # zeros = fake

            d_loss_back = d_real_error_back + d_fake_error_back
            diffopt.step(d_loss_back)
        copy_parameter(D, fmodel)
            #d_ia_optimizer.step()

    elif args.method == 'RHG' or args.method == 'CG' or args.method == 'TRHG' or args.method == 'Neumann':
        # print(args.method)
        t0 = time.time()

        d_infos = [[0, 0], [0, 0]]
        g_info, d_loss, hyper_time_bi, time_bi = loop_RHG(G, D)
        p, r, F1 = D_acc(D, torch.from_numpy(dset.sample(minibatch_size)).to(device),
                         G(torch.from_numpy(noise_sampler(minibatch_size, g_inp)).to(device)))

        timelist.append(time.time() - t0)
    else:
        assert 'Error method don\'t support'


    if it % log_interval == 0:
        timelist.append(time.time() - t0)
        g_fake_data = g_sample()
        samples.append(g_fake_data)
        
        plot_samples([g_fake_data],step=it)
        KLlist.append(KL(samples=g_fake_data))
        JSlist.append(JS(samples=g_fake_data))

        print(
            'T={}, gloss={:.4f}, F1={:.4f}, p={:.4f}, r={:.4f},time={:.4f}'.format(it, g_info, F1.item(), p.item(),
                                                                       r.item(),np.sum(np.array(timelist))))
        glist.append(g_info)
        if args.method == 'GN':
            glist_GN.append(loss_L2(g_grad_GN).item())
        dlist.append(d_loss)

        plist.append(p.item())
        rlist.append(r.item())
        F1list.append(F1.item())

        scipy.io.savemat(
            'D:/Code_2023/GDC/LwCL_results_cube/cube_result{}_act{}_noise{}_n{}_std{}_us{}_zlr{}_glr{}_dlr{}.mat'.format(args.method, args.act, args.noise,
                                                                                   args.n,
                                                                                   args.std,  args.us, args.zlr,args.glr,
                                                                                   args.dlr),
            mdict={'data': samples, 'graddata': gradlist, 'gradGN': glist_GN, 'gloss': glist, 'dloss': dlist,
                   'time': timelist, 'p': plist, 'r': rlist,
                   'F1': F1list, 'KL': KL,'JS':JS })

scipy.io.savemat(
    'D:/Code_2023/GDC/LwCL_results_cube/cube_result{}_act{}_noise{}_n{}_std{}_us{}_zlr{}_glr{}_dlr{}_{}.mat'.format(args.method, args.act, args.noise,
                                                                           args.n,
                                                                           args.std,  args.us, args.zlr,args.glr,
                                                                           args.dlr,
                                                                           time.strftime("%Y_%m_%d_%H_%M_%S")),
    mdict={'data': samples, 'graddata': gradlist, 'gradGN': glist_GN, 'gloss': glist, 'dloss': dlist,
           'time': timelist, 'p': plist, 'r': rlist,
           'F1': F1list, 'KL': KL,'JS':JS })
