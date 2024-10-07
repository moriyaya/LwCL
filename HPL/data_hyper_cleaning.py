# -*- coding: utf-8 -*-
# **********************************
# Author: Xuan Liu
# Contact:  liuxuan_16@126.com
# **********************************

import torch
import random
from collections import deque
from torchvision.datasets import MNIST,FashionMNIST,CIFAR10
import torch.nn.functional as F
import copy
import numpy as np
import time
import csv
import argparse
import higher
import numpy
import os

parser = argparse.ArgumentParser(description='Data HyperCleaner')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N')
parser.add_argument('--mode', type=str, default='', help='ours or IFT')
parser.add_argument('--hg_mode', type=str, default='GN', metavar='N',
                    help='hypergradient RHG or BDA or BDAn or TRHG,BDAn_L2')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--z_loop', type=int, default=100)
parser.add_argument('--y_loop', type=int, default=10)
parser.add_argument('--x_loop', type=int, default=3000)
parser.add_argument('--z_L2_reg', type=float, default=0.01)
parser.add_argument('--y_L2_reg', type=float, default=0.0001)
parser.add_argument('--y_L1_reg', type=float, default=0.0001)
parser.add_argument('--y_ln_reg', type=float, default=0.1)
parser.add_argument('--distance_reg', type=float, default=0.0001)
parser.add_argument('--epsilon', type=float, default=1.e-3)
parser.add_argument('--lamb', type=float, default=0.1)
parser.add_argument('--cg_steps', type=int, default=2)
parser.add_argument('--nuc_reg', type=float, default=1.e-3)
parser.add_argument('--reg_decay', type=bool, default=True)
parser.add_argument('--decay_rate', type=float, default=0.1)
parser.add_argument('--learn_h', type=bool, default=False)
parser.add_argument('--x_lr', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--alpha_GKM', type=float, default=0.1)

parser.add_argument('--y_lr', type=float, default=0.01)#0.02
parser.add_argument('--z_lr', type=float, default=0.02)
# Fahsion MNIST x lr 0.03, y lr 0.05, z lr 0.01, pretrained model y loop 200 y lr 0.03
parser.add_argument('--x_lr_decay_rate', type=float, default=0.1)
parser.add_argument('--x_lr_decay_patience', type=int, default=1)
parser.add_argument('--pollute_rate', type=float, default=0.5)
parser.add_argument('--convex', type=str, default='nonconvex')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nestrov',  action='store_true', default=False)
parser.add_argument('--GKM',  action='store_true', default=False)

parser.add_argument('--train_proxi', type=bool, default=False)
parser.add_argument('--Notes', type=str, default='5.12.nonconvex_alpha0.1CIFAR10', metavar='N', help='test_z_influence')
parser.add_argument('--l2_coef', type=float, default=0.01, help="coefficient of L2 regularization while using BDAn_new_l2")
args = parser.parse_args()
if args.GKM:
    args.y_lr=args.y_lr*args.alpha_GKM
if args.dataset=='MNIST':
    dataset = MNIST(root="./data/mnist", train=True, download=True)
elif args.dataset == 'FashionMNIST':
    dataset = FashionMNIST(root="./data/fashionmnist", train=True, download=True)
elif args.dataset=='CIFAR10':
    dataset=CIFAR10(root="./data/cifar10", train=True, download=True)
    dataset.targets=torch.from_numpy(numpy.array(dataset.targets)).long()
    dataset.data=torch.from_numpy(dataset.data)
print(args)

cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


class Dataset:
    def __init__(self, data, target, polluted=False, rho=0.0):
        self.data = data.float() / torch.max(data)
        print(list(target.shape))
        if not polluted:
            self.clean_target = target
            self.dirty_target = None
            self.clean = np.ones(list(target.shape)[0])
        else:
            self.clean_target = None
            self.dirty_target = target
            self.clean = np.zeros(list(target.shape)[0])
        self.polluted = polluted
        self.rho = rho
        self.set = set(target.numpy().tolist())

    def data_polluting(self, rho):
        assert self.polluted == False and self.dirty_target is None
        number = self.data.shape[0]
        number_list = list(range(number))
        random.shuffle(number_list)
        self.dirty_target = copy.deepcopy(self.clean_target)
        for i in number_list[:int(rho * number)]:
            dirty_set = copy.deepcopy(self.set)
            dirty_set.remove(int(self.clean_target[i]))
            self.dirty_target[i] = random.randint(0, len(dirty_set))
            self.clean[i] = 0
        self.polluted = True
        self.rho = rho

    def data_flatten(self):
        try :
            self.data = self.data.view(self.data.shape[0], self.data.shape[1] * self.data.shape[2])
        except BaseException:
            self.data = self.data.reshape(self.data.shape[0], self.data.shape[1] * self.data.shape[2] * self.data.shape[3])

    # def get_batch(self,batch_size):

    def to_cuda(self):
        self.data = self.data.to(device)
        if self.clean_target is not None:
            self.clean_target = self.clean_target.to(device)
        if self.dirty_target is not None:
            self.dirty_target = self.dirty_target.to(device)


def data_splitting(dataset, tr, val, test):
    assert tr + val + test <= 1.0 or tr > 1



    number = dataset.targets.shape[0]
    number_list = list(range(number))
    random.shuffle(number_list)
    if tr < 1:
        tr_number = tr * number
        val_number = val * number
        test_number = test * number
    else:
        tr_number = tr
        val_number = val
        test_number = test

    train_data = Dataset(dataset.data[number_list[:int(tr_number)], :, :],
                         dataset.targets[number_list[:int(tr_number)]])
    val_data = Dataset(dataset.data[number_list[int(tr_number):int(tr_number + val_number)], :, :],
                       dataset.targets[number_list[int(tr_number):int(tr_number + val_number)]])
    test_data = Dataset(
        dataset.data[number_list[int(tr_number + val_number):(tr_number + val_number + test_number)], :, :],
        dataset.targets[number_list[int(tr_number + val_number):(tr_number + val_number + test_number)]])
    return train_data, val_data, test_data

def loss_L2(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 2) ** 2
    return loss


def loss_L1(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 1)
    return loss

def distance_loss(params, init_prams):
    loss = 0
    for p,ip in zip(params,init_prams):
        loss += torch.norm(p - ip, 2)
    return loss


def loss_Lq(parameters,q,epi):
    loss = 0
    for w in parameters:
        loss += (torch.norm(w,2)+torch.norm(epi*torch.ones_like(w),2))**(q/2)
    return loss


def accuary(out, target):
    pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
    return acc


def Binarization(x):
    x_bi = np.zeros_like(x)
    for i in range(x.shape[0]):
        # print(x[i])
        x_bi[i] = 1 if x[i] >= 0 else 0
    return x_bi


def vec_to_grad(vec,model):
    pointer = 0
    res = []
    for param in model.parameters():
        num_param = param.numel()
        res.append(vec[pointer:pointer+num_param].view_as(param).data)
        pointer += num_param
    return res

def hv_prod(in_grad, x, params):
    hv = torch.autograd.grad(in_grad, params, retain_graph=True, grad_outputs=x)
    hv = torch.nn.utils.parameters_to_vector(hv).detach()
    # precondition with identity matrix
    return hv/args.lamb + x


def CG(in_grad, outer_grad, params,model,cg_steps):
    x = outer_grad.clone().detach()
    r = outer_grad.clone().detach() - hv_prod(in_grad, x, params)
    p = r.clone().detach()
    for i in range(cg_steps):
        Ap = hv_prod(in_grad, p, params)
        alpha = (r @ r) / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        r = r_new.clone().detach()
    return vec_to_grad(x,model)



def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])


class Net_x(torch.nn.Module):
    def __init__(self, tr):
        super(Net_x, self).__init__()
        self.x = torch.nn.Parameter(torch.zeros(tr.data.shape[0]).to(device).requires_grad_(True))

    def forward(self, y):
        # if torch.norm(torch.sigmoid(self.x), 1) > 2500:
        #     y = torch.sigmoid(self.x) / torch.norm(torch.sigmoid(self.x), 1) * 2500 * y
        # else:
        y = torch.sigmoid(self.x) * y
        y = y.mean()
        return y


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


tr, val, test = data_splitting(dataset, 5000, 5000, 10000)
tr.data_polluting(args.pollute_rate)
tr.data_flatten()
val.data_flatten()
test.data_flatten()
tr.to_cuda()
val.to_cuda()
test.to_cuda()
log_path = "{}_hg_mode{}_network{}_outerLoop{}_inner_loop{}_outer_lr{}_inner_lr{}_z_lr{}_l2_coef{}_pollute_rate{}_Nestrov{}_Notes{}.csv".format(args.dataset,args.hg_mode,args.convex,args.x_loop,args.y_loop,args.x_lr,args.y_lr,args.z_lr,args.l2_coef,args.pollute_rate,args.nestrov, args.Notes)
with open(log_path, 'a', encoding='utf-8') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([args])
    csv_writer.writerow([args.hg_mode, 'acc', 'F1 score','total_time_iter','forward_time','backward_time','val loss','p','r','pmax_list'])
d = 28 ** 2
n = 10
z_loop = args.z_loop
y_loop = args.y_loop
x_loop = args.x_loop

# x = torch.zeros(tr.data.shape[0]).to("cuda").requires_grad_(True)
x = Net_x(tr)
if args.dataset=='MNIST' or args.dataset=='FashionMNIST':
    d = 28 ** 2
    n = 10
    if args.convex=='convex':
        y = torch.nn.Sequential(torch.nn.Linear(d, n)).to(device)
        y_backup = torch.nn.Sequential(torch.nn.Linear(d, n)).to(device)
        y_proxi = torch.nn.Sequential(torch.nn.Linear(d, n)).to(device)
# non-convex
    else:
        y = torch.nn.Sequential(torch.nn.Linear(d, 300), torch.nn.Linear(300, n)).to(device)
        y_backup = torch.nn.Sequential(torch.nn.Linear(d, 300), torch.nn.Linear(300, n)).to(device)
        y_proxi = torch.nn.Sequential(torch.nn.Linear(d, 300), torch.nn.Linear(300, n)).to(device)
elif args.dataset=='CIFAR10':
    d=3*32**2
    n=10
    if args.convex=='convex':
        y = torch.nn.Sequential(torch.nn.Linear(d, n)).to(device)
        y_backup = torch.nn.Sequential(torch.nn.Linear(d, n)).to(device)
        y_proxi = torch.nn.Sequential(torch.nn.Linear(d, n)).to(device)
    else:
        y = torch.nn.Sequential(torch.nn.Linear(d, 300), torch.nn.Linear(300, n)).to(device)
        y_backup = torch.nn.Sequential(torch.nn.Linear(d, 300), torch.nn.Linear(300, n)).to(device)
        y_proxi = torch.nn.Sequential(torch.nn.Linear(d, 300), torch.nn.Linear(300, n)).to(device)
#convex
# y = torch.nn.Sequential(torch.nn.Linear(d, n)).to("cuda")
# z = torch.nn.Sequential(torch.nn.Linear(d, n)).to("cuda")

x_opt = torch.optim.Adam(x.parameters(), lr=args.x_lr)
if args.nestrov:
    y_opt = torch.optim.SGD(y.parameters(), lr=args.y_lr,momentum=args.momentum, nesterov=True)
else:
    y_opt=torch.optim.SGD(y.parameters(), lr=args.y_lr) 
acc_history = []
clean_acc_history = []
print(sum(p.numel() for p in y.parameters()))
Lq=1/sum(p.numel() for p in y.parameters())
loss_x_l = 0
F1_score_last=0
lr_decay_rate = 1
reg_decay_rate = 1
dc=0


def nuclear_norm_loss(input_params, target, lambda_nn=1e-3):
    x = input[:, :, 1:, :] - input[:, :, :-1, :]
    y = input[:, :, :, 1:] - input[:, :, :, :-1]

    delta_x = x[:, :, 1:, :-2] ** 2
    delta_y = y[:, :, :-2, 1:] ** 2
    delta_u = torch.abs(delta_x + delta_y)

    nn = torch.norm(delta_u.flatten(1, -1), p='nuc')
    loss = lambda_nn * nn

    return loss


 
total_time = 0
if args.train_proxi:
    y_opt = torch.optim.SGD(y_proxi.parameters(), lr=args.y_lr)
    acc_list=[]
    for iter in range(5000):
        y_opt.zero_grad()
        loss=F.cross_entropy(y_proxi(tr.data),tr.clean_target)
        loss.backward()
        y_opt.step()
        with torch.no_grad():
            out = y_proxi(test.data)
            acc = accuary(out, test.clean_target)
            acc_list.append(acc)
            print("iter {}, test accuracy {}".format(iter,acc))
        if acc_list.index(max(acc_list)) == iter:
            print('Checkpoint Updated!Acc:' +str(acc))
            torch.save(y_proxi,'y_proxi'+args.dataset+'.pt')

else:
    for x_itr in range(x_loop):
        # x_opt.param_groups[0]['lr']=args.x_lr/(1+(1e-5)*(x_itr+1))

        for xp in x.parameters():
            if xp.grad is not None:
                xp.grad = None
        for yp in y.parameters():
            if yp.grad is not None:
                yp.grad = None
        # print(loss_L1(y.parameters()).item())
        start_time_task = time.time()

        forward_time, backward_time = 0, 0


        if args.hg_mode == 'RHG' or args.hg_mode == 'BDA':
            u=args.alpha
            with higher.innerloop_ctx(y, y_opt,copy_initial_weights=False) as (fmodel, diffopt):
                pmax=0
                forward_time_task = time.time()
                for y_idx in range(y_loop):
                    out_f = fmodel(tr.data)

                    if args.hg_mode == 'RHG' or args.hg_mode == 'TRHG':
                        loss_f = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                        # print(loss_f.item())
                    elif args.hg_mode == 'BDA':
                        out_F = fmodel(val.data)
                        alpha = 1. / np.sqrt(y_idx + 1)
                        loss_f =(1-u)* x(F.cross_entropy(out_f, tr.dirty_target, reduction='none')) +  alpha *u * F.cross_entropy(out_F, val.clean_target)
                    diffopt.step(loss_f)
                    # print(loss_L1(fmodel.parameters()))
                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()
                out_F = fmodel(val.data)
                loss_F = F.cross_entropy(out_F, val.clean_target)
                # print(loss_F.item())
                # print(accuary(fmodel(test.data), test.clean_target))
                grad_x = torch.autograd.grad(loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
                backward_time_task = time.time() - backward_time_task
                backward_time += backward_time_task
                total_time_iter=time.time() - start_time_task
                total_time = total_time + total_time_iter
                x_opt.step()
                copy_parameter(y,fmodel)
            if args.hg_mode == 'GN':
                # with higher.innerloop_ctx(y, y_opt, copy_initial_weights=False) as (fmodel, diffopt):
                pmax = 0
                x_opt.zero_grad()

                forward_time_task = time.time()
                for y_idx in range(y_loop):
                    y_opt.zero_grad()
                    out_f = y(tr.data)

                    loss_f = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                        # print(loss_f.item())
                    y_opt.step(loss_f)
                    # print(loss_L1(fmodel.parameters()))
                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()
                out_F = y(val.data)
                loss_F = F.cross_entropy(out_F, val.clean_target)
                gFyfy = 0   

                dFyA = torch.autograd.grad(loss_F, y.parameters(), retain_graph=True)
                dfyA = torch.autograd.grad(loss_f, y.parameters(), retain_graph=True)

                for Fy, fy in zip(dFyA, dfyA):
                    gFyfy = gFyfy + torch.sum(Fy * fy)
                    gfyfy = gfyfy + torch.sum(fy * fy) + 1e-10 
                GN_loss = -gFyfy.detach() / gfyfy.detach() * f
                GN_loss.backward()
  
                loss_F.backward()

                backward_time_task = time.time() - backward_time_task
                backward_time += backward_time_task
                total_time_iter = time.time() - start_time_task
                total_time = total_time + total_time_iter
                x_opt.step()
                copy_parameter(y, fmodel)
            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc = 0
                    if F1_score_last > F1_score:
                        dc = 1
                    F1_score_last = F1_score

                    # x_opt_l.step(F1_score)
                    # y_opt_l.step(acc)
                    # z_opt_l.step(acc)
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f}'.format(x_itr,
                                                                                                                100 * accuary(
                                                                                                                    out,
                                                                                                                test.clean_target),
                                                                                                                100 * p,
                                                                                                                100 * r,
                                                                                                                100 * F1_score,
                                                                                                                loss_F))
                with open(log_path, 'a', encoding='utf-8', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(
                        [x_itr, 100. * acc, 100. * F1_score, total_time_iter, forward_time, backward_time, loss_F.item(),
                         100 * p, 100 * r, pmax,total_time])
        elif args.hg_mode=='BDAn':
            F_list = []
            with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
                forward_time_task = time.time()
 
                for y_idx in range(args.y_loop):
                    out_f = fmodel(tr.data)

                    # print(loss_Lq(fmodel.parameters(),0.5,0.1))
                    loss_f = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                    #grad_y=torch.autograd.grad(loss_f,fmodel.parameters(),retain_graph=True)
                    total_loss_f= loss_f #+ args.y_L2_reg*loss_L2(grad_y)
                    diffopt.step(total_loss_f)
                    F_list.append(F.cross_entropy(fmodel(val.data), val.clean_target).item())
                pmax = F_list.index(max(F_list))
                #pmax_buffer.append(pmax)
                print(pmax)
                out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))

                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()
                #out_F = fmodel(val.data, params=fmodel.parameters())
                loss_F = F.cross_entropy(out_F, val.clean_target)
                grad_z = torch.autograd.grad(loss_F, fmodel.parameters(time=0), retain_graph=True)
                grad_x = torch.autograd.grad(loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
                    # print(p.data)
                for p, yp in zip(grad_z, y.parameters()):
                    if yp.grad == None:
                        yp.grad = p
                    else:
                        yp.grad += p
                copy_parameter(y_backup,fmodel)

            backward_time_task = time.time() - backward_time_task
            backward_time += backward_time_task
            total_time_iter=time.time() - start_time_task
            total_time = total_time + total_time_iter
            y_opt.step()
            x_opt.step()

            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y_backup(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc=0
                    if F1_score_last>F1_score:
                        dc=1
                    F1_score_last=F1_score

                    # x_opt_l.step(F1_score)
                    # y_opt_l.step(acc)
                    # z_opt_l.step(acc)
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax={}'.format(x_itr,
                                                                                         100 * accuary(out,
                                                                                                       test.clean_target),
                                                                                         100 * p, 100 * r, 100 * F1_score,loss_F,pmax))

                    with open(log_path, 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([x_itr, 100. * acc, 100. * F1_score,total_time_iter,forward_time,backward_time,loss_F.item(),100*p,100*r,pmax,total_time])
        elif args.hg_mode=='BDAn_l2':
            F_list = []
            y_proxi=torch.load('y_proxi'+args.dataset+'.pt')
            with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
                forward_time_task = time.time()
 

                for y_idx in range(args.y_loop):
                    out_f = fmodel(tr.data)

                    # print(loss_Lq(fmodel.parameters(),0.5,0.1))
                    loss_f = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                    #grad_y=torch.autograd.grad(loss_f,fmodel.parameters(),retain_graph=True)
                    total_loss_f= loss_f #+ args.y_L2_reg*loss_L2(grad_y)
                    diffopt.step(total_loss_f)
                    if x_itr ==0:
                        coef=1.0
                    else:
                        coef=0.0
                    reg_l2=distance_loss(fmodel.parameters(time=0),y_proxi.parameters())
                    F_loss=F.cross_entropy(fmodel(val.data), val.clean_target)+ coef*args.l2_coef*reg_l2
                    F_list.append(F_loss.item())
                pmax = F_list.index(max(F_list))
                #pmax = args.y_loop-1
                #pmax_buffer.append(pmax)
                print(pmax)
                out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))

                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()
                #out_F = fmodel(val.data, params=fmodel.parameters())
                if x_itr == 0:
                    coef = 1.0
                else:
                    coef = 0.0
                reg_l2 = distance_loss(fmodel.parameters(time=0), y_proxi.parameters())
                loss_F = F.cross_entropy(out_F, val.clean_target) +coef*args.l2_coef*reg_l2
                grad_z = torch.autograd.grad(loss_F, fmodel.parameters(time=0), retain_graph=True)
                grad_x = torch.autograd.grad(loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
                    # print(p.data)
                for p, yp in zip(grad_z, y.parameters()):
                    if yp.grad == None:
                        yp.grad = p
                    else:
                        yp.grad += p
                copy_parameter(y_backup,fmodel)

            backward_time_task = time.time() - backward_time_task
            backward_time += backward_time_task
            total_time_iter=time.time() - start_time_task
            total_time = total_time + total_time_iter
            y_opt.step()
            x_opt.step()

            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y_backup(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc=0
                    if F1_score_last>F1_score:
                        dc=1
                    F1_score_last=F1_score

                    # x_opt_l.step(F1_score)
                    # y_opt_l.step(acc)
                    # z_opt_l.step(acc)
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax={}'.format(x_itr,
                                                                                         100 * accuary(out,
                                                                                                       test.clean_target),
                                                                                         100 * p, 100 * r, 100 * F1_score,loss_F,pmax))

                    with open(log_path, 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([x_itr, 100. * acc, 100. * F1_score,total_time_iter,forward_time,backward_time,loss_F.item(),100*p,100*r,pmax,total_time])
        elif args.hg_mode=='BDAn_IA':
            F_list = []
            with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
                forward_time_task = time.time()
 

                for y_idx in range(args.y_loop):
                    out_f = fmodel(tr.data)

                    # print(loss_Lq(fmodel.parameters(),0.5,0.1))
                    loss_f = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                    #grad_y=torch.autograd.grad(loss_f,fmodel.parameters(),retain_graph=True)
                    total_loss_f= loss_f #+ args.y_L2_reg*loss_L2(grad_y)
                    diffopt.step(total_loss_f)
                    F_list.append(F.cross_entropy(fmodel(val.data), val.clean_target).item())
                #pmax = F_list.index(max(F_list))
                pmax = args.y_loop -1
                #pmax_buffer.append(pmax)
                print(pmax)
                out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))

                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()
                #out_F = fmodel(val.data, params=fmodel.parameters())
                loss_F = F.cross_entropy(out_F, val.clean_target)
                grad_z = torch.autograd.grad(loss_F, fmodel.parameters(time=0), retain_graph=True)
                grad_x = torch.autograd.grad(loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
                    # print(p.data)
                for p, yp in zip(grad_z, y.parameters()):
                    if yp.grad == None:
                        yp.grad = p
                    else:
                        yp.grad += p
                copy_parameter(y_backup,fmodel)

            backward_time_task = time.time() - backward_time_task
            backward_time += backward_time_task
            total_time_iter=time.time() - start_time_task
            total_time = total_time + total_time_iter
            y_opt.step()
            x_opt.step()

            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y_backup(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc=0
                    if F1_score_last>F1_score:
                        dc=1
                    F1_score_last=F1_score

 
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax={}'.format(x_itr,
                                                                                         100 * accuary(out,
                                                                                                       test.clean_target),
                                                                                         100 * p, 100 * r, 100 * F1_score,loss_F,pmax))

                    with open(log_path, 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([x_itr, 100. * acc, 100. * F1_score,total_time_iter,forward_time,backward_time,loss_F.item(),100*p,100*r,pmax,total_time])
        elif args.hg_mode=='BDAn_IA_l2':
            F_list = []
            y_proxi=torch.load('y_proxi'+args.dataset+'.pt')
            with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
                forward_time_task = time.time()
 

                for y_idx in range(args.y_loop):
                    out_f = fmodel(tr.data)

                    # print(loss_Lq(fmodel.parameters(),0.5,0.1))
                    loss_f = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                    #grad_y=torch.autograd.grad(loss_f,fmodel.parameters(),retain_graph=True)
                    total_loss_f= loss_f #+ args.y_L2_reg*loss_L2(grad_y)
                    diffopt.step(total_loss_f)
                    if x_itr ==0:
                        coef=1.0
                    else:
                        coef=0.0
                    reg_l2=distance_loss(fmodel.parameters(time=0),y_proxi.parameters())
                    F_loss=F.cross_entropy(fmodel(val.data), val.clean_target)+ coef*args.l2_coef*reg_l2
                    F_list.append(F_loss.item())
                #pmax = F_list.index(max(F_list))
                pmax = args.y_loop-1
                #pmax_buffer.append(pmax)
                print(pmax)
                out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))

                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()
                #out_F = fmodel(val.data, params=fmodel.parameters())
                if x_itr == 0:
                    coef = 1.0
                else:
                    coef = 0.0
                reg_l2 = distance_loss(fmodel.parameters(time=0), y_proxi.parameters())
                loss_F = F.cross_entropy(out_F, val.clean_target) +coef*args.l2_coef*reg_l2
                grad_z = torch.autograd.grad(loss_F, fmodel.parameters(time=0), retain_graph=True)
                grad_x = torch.autograd.grad(loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
                    # print(p.data)
                for p, yp in zip(grad_z, y.parameters()):
                    if yp.grad == None:
                        yp.grad = p
                    else:
                        yp.grad += p
                copy_parameter(y_backup,fmodel)

            backward_time_task = time.time() - backward_time_task
            backward_time += backward_time_task
            total_time_iter=time.time() - start_time_task
            total_time = total_time + total_time_iter
            y_opt.step()
            x_opt.step()

            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y_backup(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc=0
                    if F1_score_last>F1_score:
                        dc=1
                    F1_score_last=F1_score

 
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax={}'.format(x_itr,
                                                                                         100 * accuary(out,
                                                                                                       test.clean_target),
                                                                                         100 * p, 100 * r, 100 * F1_score,loss_F,pmax))

                    with open(log_path, 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([x_itr, 100. * acc, 100. * F1_score,total_time_iter,forward_time,backward_time,loss_F.item(),100*p,100*r,pmax,total_time])
        elif args.hg_mode == 'BDA_IAPTT':
            F_list = []
            u = args.alpha
            with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
                forward_time_task = time.time()
 

                for y_idx in range(args.y_loop):
                    out_f = fmodel(tr.data)

                    out_F = fmodel(val.data)
                    alpha = 1. / np.sqrt(y_idx + 1)
                    loss_f = (1 - u) * x(F.cross_entropy(out_f, tr.dirty_target, reduction='none')) + alpha *u * F.cross_entropy(out_F, val.clean_target)
 
                    diffopt.step(loss_f)
                    F_list.append(F.cross_entropy(fmodel(val.data), val.clean_target).item())
                pmax = F_list.index(max(F_list))
                # pmax_buffer.append(pmax)
                print(pmax)
                out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))

                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()
                # out_F = fmodel(val.data, params=fmodel.parameters())
                loss_F = F.cross_entropy(out_F, val.clean_target)
                grad_z = torch.autograd.grad(loss_F, fmodel.parameters(time=0), retain_graph=True)
                grad_x = torch.autograd.grad(loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
                    # print(p.data)
                for p, yp in zip(grad_z, y.parameters()):
                    if yp.grad == None:
                        yp.grad = p
                    else:
                        yp.grad += p
                copy_parameter(y_backup,fmodel)
                # for param_orig, param_new in zip(y_backup.parameters(),fmodel.parameters()):
                #     param_orig = param_new.clone.detach()
            backward_time_task = time.time() - backward_time_task
            backward_time += backward_time_task
            total_time_iter = time.time() - start_time_task
            y_opt.step()
            x_opt.step()

            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y_backup(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc = 0
                    if F1_score_last > F1_score:
                        dc = 1
                    F1_score_last = F1_score

                    # x_opt_l.step(F1_score)
                    # y_opt_l.step(acc)
                    # z_opt_l.step(acc)
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax={}'.format(x_itr,
                                                                                                                 100 * accuary(
                                                                                                                     out,
                                                                                                                     test.clean_target),
                                                                                                                 100 * p,
                                                                                                                 100 * r,
                                                                                                                 100 * F1_score,
                                                                                                                 loss_F,
                                                                                                                 pmax))

                    with open(log_path, 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(
                            [x_itr, 100. * acc, 100. * F1_score, total_time_iter, forward_time, backward_time,
                             loss_F.item(), 100 * p, 100 * r, pmax])
        elif args.hg_mode == 'BDA_IA':
            F_list = []
            u = args.alpha
            with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
                forward_time_task = time.time()
   

                for y_idx in range(args.y_loop):
                    out_f = fmodel(tr.data)

                    out_F = fmodel(val.data)
                    alpha = 1. / np.sqrt(y_idx + 1)
                    loss_f = (1 - u) * x(F.cross_entropy(out_f, tr.dirty_target, reduction='none')) + alpha *u * F.cross_entropy(out_F, val.clean_target)
    
                    diffopt.step(loss_f)
                    F_list.append(F.cross_entropy(fmodel(val.data), val.clean_target).item())
                #pmax = F_list.index(max(F_list))
                pmax=args.y_loop-1
                # pmax_buffer.append(pmax)
                print(pmax)
                out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))

                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()
                # out_F = fmodel(val.data, params=fmodel.parameters())
                loss_F = F.cross_entropy(out_F, val.clean_target)
                grad_z = torch.autograd.grad(loss_F, fmodel.parameters(time=0), retain_graph=True)
                grad_x = torch.autograd.grad(loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
                    # print(p.data)
                for p, yp in zip(grad_z, y.parameters()):
                    if yp.grad == None:
                        yp.grad = p
                    else:
                        yp.grad += p
                copy_parameter(y_backup,fmodel)
                # for param_orig, param_new in zip(y_backup.parameters(),fmodel.parameters()):
                #     param_orig = param_new.clone.detach()
            backward_time_task = time.time() - backward_time_task
            backward_time += backward_time_task
            total_time_iter = time.time() - start_time_task
            y_opt.step()
            x_opt.step()

            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y_backup(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc = 0
                    if F1_score_last > F1_score:
                        dc = 1
                    F1_score_last = F1_score

                    # x_opt_l.step(F1_score)
                    # y_opt_l.step(acc)
                    # z_opt_l.step(acc)
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax={}'.format(x_itr,
                                                                                                                 100 * accuary(
                                                                                                                     out,
                                                                                                                     test.clean_target),
                                                                                                                 100 * p,
                                                                                                                 100 * r,
                                                                                                                 100 * F1_score,
                                                                                                                 loss_F,
                                                                                                                 pmax))

                    with open(log_path, 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(
                            [x_itr, 100. * acc, 100. * F1_score, total_time_iter, forward_time, backward_time,
                             loss_F.item(), 100 * p, 100 * r, pmax])
        elif args.hg_mode == 'BDA_IA_l2':
            F_list = []
            y_proxi=torch.load('y_proxi'+args.dataset+'.pt')
            u = args.alpha
            with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
                forward_time_task = time.time()
 

                for y_idx in range(args.y_loop):
                    out_f = fmodel(tr.data)

                    out_F = fmodel(val.data)
                    alpha = 1. / np.sqrt(y_idx + 1)
                    if x_itr ==0:
                        coef=1.0
                    else:
                        coef=0.0
                    reg_l2=distance_loss(fmodel.parameters(time=0),y_proxi.parameters())
                    loss_f = (1 - u) * x(F.cross_entropy(out_f, tr.dirty_target, reduction='none')) + alpha *u *( F.cross_entropy(out_F, val.clean_target)+ coef*args.l2_coef*reg_l2)
 
                    diffopt.step(loss_f)
                    F_list.append(F.cross_entropy(fmodel(val.data), val.clean_target).item())
                #pmax = F_list.index(max(F_list))
                pmax=args.y_loop-1
                # pmax_buffer.append(pmax)
                print(pmax)
                out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))

                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()
                # out_F = fmodel(val.data, params=fmodel.parameters())
                if x_itr == 0:
                    coef = 1.0
                else:
                    coef = 0.0
                reg_l2 = distance_loss(fmodel.parameters(time=0), y_proxi.parameters())
                loss_F = F.cross_entropy(out_F, val.clean_target)+ coef*args.l2_coef*reg_l2
                grad_z = torch.autograd.grad(loss_F, fmodel.parameters(time=0), retain_graph=True)
                grad_x = torch.autograd.grad(loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
                    # print(p.data)
                for p, yp in zip(grad_z, y.parameters()):
                    if yp.grad == None:
                        yp.grad = p
                    else:
                        yp.grad += p
                copy_parameter(y_backup,fmodel)
                # for param_orig, param_new in zip(y_backup.parameters(),fmodel.parameters()):
                #     param_orig = param_new.clone.detach()
            backward_time_task = time.time() - backward_time_task
            backward_time += backward_time_task
            total_time_iter = time.time() - start_time_task
            y_opt.step()
            x_opt.step()

            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y_backup(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc = 0
                    if F1_score_last > F1_score:
                        dc = 1
                    F1_score_last = F1_score

                    # x_opt_l.step(F1_score)
                    # y_opt_l.step(acc)
                    # z_opt_l.step(acc)
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax={}'.format(x_itr,
                                                                                                                 100 * accuary(
                                                                                                                     out,
                                                                                                                     test.clean_target),
                                                                                                                 100 * p,
                                                                                                                 100 * r,
                                                                                                                 100 * F1_score,
                                                                                                                 loss_F,
                                                                                                                 pmax))

                    with open(log_path, 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(
                            [x_itr, 100. * acc, 100. * F1_score, total_time_iter, forward_time, backward_time,
                             loss_F.item(), 100 * p, 100 * r, pmax])
        elif args.hg_mode=='BDAn_nuclear':
            F_list = []
            with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
                forward_time_task = time.time()
 

                for y_idx in range(args.y_loop):
                    out_f = fmodel(tr.data)
                    # print(loss_Lq(fmodel.parameters(),0.5,0.1))
                    loss_f = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                    if y_idx>0:
                        out_f_pre = fmodel(tr.data,params=fmodel.parameters(time=0))
                        total_loss_f = loss_f + args.nuc_reg * torch.norm(out_f - out_f_pre,'nuc')  # + args.y_L2_reg*loss_L2(grad_y)
                    else:
                        total_loss_f = loss_f
                    #grad_y=torch.autograd.grad(loss_f,fmodel.parameters(),retain_graph=True)
                    diffopt.step(total_loss_f)
                    F_list.append(F.cross_entropy(fmodel(val.data), val.clean_target).item())
                pmax = F_list.index(max(F_list))
                #pmax_buffer.append(pmax)
                print(pmax)
                out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))
                #out_F_pre_model = fmodel(val.data, params=fmodel.parameters(time=0))
                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()
                #out_F = fmodel(val.data, params=fmodel.parameters())
                loss_F = F.cross_entropy(out_F, val.clean_target)
                grad_z = torch.autograd.grad(loss_F, fmodel.parameters(time=0), retain_graph=True)
                grad_x = torch.autograd.grad(loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
                    # print(p.data)
                for p, yp in zip(grad_z, y.parameters()):
                    if yp.grad == None:
                        yp.grad = p
                    else:
                        yp.grad += p

            backward_time_task = time.time() - backward_time_task
            backward_time += backward_time_task
            total_time_iter=time.time() - start_time_task
            y_opt.step()
            x_opt.step()

            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc=0
                    if F1_score_last>F1_score:
                        dc=1
                    F1_score_last=F1_score

 
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax={}'.format(x_itr,
                                                                                         100 * accuary(out,
                                                                                                       test.clean_target),
                                                                                         100 * p, 100 * r, 100 * F1_score,loss_F,pmax))

                    with open(log_path, 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([x_itr, 100. * acc, 100. * F1_score,total_time_iter,forward_time,backward_time,loss_F.item(),100*p,100*r,pmax])
        elif args.hg_mode=='BDAn_epislon':
            F_list = []
            with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
                forward_time_task = time.time()
 

                for y_idx in range(args.y_loop):
                    out_f = fmodel(tr.data)

                    # print(loss_Lq(fmodel.parameters(),0.5,0.1))
                    loss_f = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                    #grad_y=torch.autograd.grad(loss_f,fmodel.parameters(),retain_graph=True)
                    total_loss_f= loss_f #+ args.y_L2_reg*loss_L2(grad_y)
                    diffopt.step(total_loss_f)
                    F_list.append(F.cross_entropy(fmodel(val.data), val.clean_target).item())
                pmax_orig = F_list.index(max(F_list))

                if pmax_orig == 0:
                    pmax = pmax_orig
                else:
                    pmax = pmax_orig - 1
                    for i in range(args.y_loop - 1):
                        if abs(F_list[pmax_orig] - F_list[pmax]) / abs(
                                F_list[pmax_orig]) < args.epsilon and pmax > 0:
                            pmax = pmax - 1
                        else:
                            break

                if x_loop % 20 == 0:
                    print(
                        "max_iter_orig =" + str(pmax_orig) + ", Loss = " + str(F_list[pmax_orig]) + " max_epislon =" + str(
                            pmax) + ", Loss = " + str(F_list[pmax]))

                #pmax_buffer.append(pmax)
                out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))

                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()
                #out_F = fmodel(val.data, params=fmodel.parameters())
                loss_F = F.cross_entropy(out_F, val.clean_target)
                grad_z = torch.autograd.grad(loss_F, fmodel.parameters(time=0), retain_graph=True)
                grad_x = torch.autograd.grad(loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
                    # print(p.data)
                for p, yp in zip(grad_z, y.parameters()):
                    if yp.grad == None:
                        yp.grad = p
                    else:
                        yp.grad += p

            backward_time_task = time.time() - backward_time_task
            backward_time += backward_time_task
            total_time_iter=time.time() - start_time_task
            y_opt.step()
            x_opt.step()

            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc=0
                    if F1_score_last>F1_score:
                        dc=1
                    F1_score_last=F1_score

 
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax={}'.format(x_itr,
                                                                                         100 * accuary(out,
                                                                                                       test.clean_target),
                                                                                         100 * p, 100 * r, 100 * F1_score,loss_F,pmax))

                    with open(log_path, 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([x_itr, 100. * acc, 100. * F1_score,total_time_iter,forward_time,backward_time,loss_F.item(),100*p,100*r,pmax])
        elif args.hg_mode=='BDAnimplicit':
            F_list = []
            alpha = args.alpha
            with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
                forward_time_task = time.time()
 

                for y_idx in range(args.y_loop):
                    out_f = fmodel(tr.data)
                    out_F = fmodel(val.data)
                    # loss_f = (1 - alpha) * x(
                    #     F.cross_entropy(out_f, tr.dirty_target, reduction='none')) + alpha * F.cross_entropy(out_F,
                    #                                                                                          val.clean_target)
                    if y_idx > 0:
                        distance_y = distance_loss(fmodel.parameters(time=y_idx),fmodel.parameters(time=0))
                        # print(loss_Lq(fmodel.parameters(),0.5,0.1))
                        loss_f = (1 - alpha) * x(
                        F.cross_entropy(out_f, tr.dirty_target, reduction='none')) + alpha * F.cross_entropy(out_F,
                                                                                                             val.clean_target) + args.distance_reg*distance_y
                    else:
                        loss_f = (1 - alpha) * x(
                        F.cross_entropy(out_f, tr.dirty_target, reduction='none')) + alpha * F.cross_entropy(out_F,
                                                                                                             val.clean_target)
                    #grad_y=torch.autograd.grad(loss_f,fmodel.parameters(),retain_graph=True)
                    total_loss_f= loss_f #+ args.y_L2_reg*loss_L2(grad_y)
                    diffopt.step(total_loss_f)
                    F_list.append(F.cross_entropy(fmodel(val.data), val.clean_target).item())
                pmax = F_list.index(max(F_list))
                #pmax_buffer.append(pmax)
                print(pmax)
                out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))

                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()

                out_f = fmodel(tr.data, params=fmodel.parameters(time=pmax + 1))

                # print(loss_Lq(fmodel.parameters(),0.5,0.1))
                loss_f = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                loss_F = F.cross_entropy(out_F, val.clean_target)
                inner_grad = torch.nn.utils.parameters_to_vector(
                    torch.autograd.grad(loss_f, fmodel.parameters(time=pmax + 1), create_graph=True))
                outer_grad = torch.nn.utils.parameters_to_vector(
                    torch.autograd.grad(loss_F, fmodel.parameters(time=pmax + 1), retain_graph=True))
                grad_z = CG(inner_grad, outer_grad, list(fmodel.parameters(time=pmax + 1)), y, args.cg_steps)

                #out_F = fmodel(val.data, params=fmodel.parameters())

                #grad_z = torch.autograd.grad(loss_F, fmodel.parameters(time=0), retain_graph=True)
                grad_x = torch.autograd.grad(loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
                    # print(p.data)
                for p, yp in zip(grad_z, y.parameters()):
                    if yp.grad == None:
                        yp.grad = p
                    else:
                        yp.grad += p

            backward_time_task = time.time() - backward_time_task
            backward_time += backward_time_task
            total_time_iter=time.time() - start_time_task
            y_opt.step()
            x_opt.step()

            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc=0
                    if F1_score_last>F1_score:
                        dc=1
                    F1_score_last=F1_score

                    # x_opt_l.step(F1_score)
                    # y_opt_l.step(acc)
                    # z_opt_l.step(acc)
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax={}'.format(x_itr,
                                                                                         100 * accuary(out,
                                                                                                       test.clean_target),
                                                                                         100 * p, 100 * r, 100 * F1_score,loss_F,pmax))

                    with open(log_path, 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([x_itr, 100. * acc, 100. * F1_score,total_time_iter,forward_time,backward_time,loss_F.item(),100*p,100*r,pmax])
        elif args.hg_mode=='BDAnmaxtopk':
            F_list = []
            with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
                forward_time_task = time.time()
 
                loss_f_list=[]
                for y_idx in range(args.y_loop):
                    out_f = fmodel(tr.data)

                    # print(loss_Lq(fmodel.parameters(),0.5,0.1))
                    loss_f = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                    # grad_y=torch.autograd.grad(loss_f,fmodel.parameters(),retain_graph=True)
                    total_loss_f = loss_f  # + args.y_L2_reg*loss_L2(grad_y)
                    loss_f_list.append(float(loss_f.cpu().detach()))
                    diffopt.step(total_loss_f)
                    F_list.append(F.cross_entropy(fmodel(val.data), val.clean_target).cpu().detach())
                #pmax_orig = F_list.index(max(F_list))
                print(loss_f_list)
                set_new= [float(F_list[i]) for i in range(len(F_list))]
                #print(set_new[0])
                F_set = frozenset(set_new)
                sorted_F_set = sorted(F_set, reverse=True)
                thedict = {}
                for j in range(3):
                    positions = [i for i, x in enumerate(F_list) if x == sorted_F_set[j]]
                    thedict[sorted_F_set[j]] = positions
                if x_loop % 10 == 0:
                    print('1st = ' + str(sorted_F_set[0]) + ' at ' + str(thedict.get(sorted_F_set[0])))

                    print('2nd = ' + str(sorted_F_set[1]) + ' at ' + str(thedict.get(sorted_F_set[1])))

                    print('3rd = ' + str(sorted_F_set[2]) + ' at ' + str(thedict.get(sorted_F_set[2])))

                pmax_list = thedict.get(sorted_F_set[0]) + thedict.get(sorted_F_set[1]) + thedict.get(sorted_F_set[2])
                pmax = min(pmax_list)
                print(pmax_list)
                print("max position is " + str(pmax))

                # pmax_buffer.append(pmax)
                out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))

                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()
                # out_F = fmodel(val.data, params=fmodel.parameters())
                loss_F = F.cross_entropy(out_F, val.clean_target)
                grad_z = torch.autograd.grad(loss_F, fmodel.parameters(time=0), retain_graph=True)
                grad_x = torch.autograd.grad(loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
                    # print(p.data)
                for p, yp in zip(grad_z, y.parameters()):
                    if yp.grad == None:
                        yp.grad = p
                    else:
                        yp.grad += p

            backward_time_task = time.time() - backward_time_task
            backward_time += backward_time_task
            total_time_iter = time.time() - start_time_task
            y_opt.step()
            x_opt.step()

            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc = 0
                    if F1_score_last > F1_score:
                        dc = 1
                    F1_score_last = F1_score
 
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax={}'.format(x_itr,
                                                                                                                 100 * accuary(
                                                                                                                     out,
                                                                                                                     test.clean_target),
                                                                                                                 100 * p,
                                                                                                                 100 * r,
                                                                                                                 100 * F1_score,
                                                                                                                 loss_F,
                                                                                                                 pmax))

                    with open(log_path, 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(
                            [x_itr, 100. * acc, 100. * F1_score, total_time_iter, forward_time, backward_time,
                             loss_F.item(), 100 * p, 100 * r, pmax])
        elif args.hg_mode=='BDAn_L2':
            F_list = []
            with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
                forward_time_task = time.time()
 

                for y_idx in range(args.y_loop):
                    out_f = fmodel(tr.data)

                    # print(loss_Lq(fmodel.parameters(),0.5,0.1))
                    loss_f = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                    total_loss_f = loss_f  # + args.y_L2_reg*loss_L2(grad_y)
                    diffopt.step(loss_f)
                    out_f = fmodel(tr.data)
                    # print(loss_Lq(fmodel.parameters(),0.5,0.1))
                    loss_f_new = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                    grad_y = torch.autograd.grad(loss_f_new, fmodel.parameters(), retain_graph=True)
                    F_list.append(
                        (F.cross_entropy(fmodel(val.data), val.clean_target) + args.y_L2_reg * loss_L2(grad_y)).item())

                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()

                pmax = F_list.index(max(F_list))

                #pmax_buffer.append(pmax)
                print(pmax)
                out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))

                #out_F = fmodel(val.data, params=fmodel.parameters())
                loss_F = F.cross_entropy(out_F, val.clean_target)
                out_f_new=fmodel(tr.data, params=fmodel.parameters(time=pmax + 1))
                loss_f_new = x(F.cross_entropy(out_f_new, tr.dirty_target, reduction='none'))
                grad_y=torch.autograd.grad(loss_f_new,fmodel.parameters(time=pmax + 1),retain_graph=True)
                total_loss_F = loss_F  + args.y_L2_reg*loss_L2(grad_y)
                grad_z = torch.autograd.grad(total_loss_F, fmodel.parameters(time=0), retain_graph=True)
                grad_x = torch.autograd.grad(total_loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
                    # print(p.data)
                for p, yp in zip(grad_z, y.parameters()):
                    if yp.grad == None:
                        yp.grad = p
                    else:
                        yp.grad += p

            backward_time_task = time.time() - backward_time_task
            backward_time += backward_time_task
            total_time_iter=time.time() - start_time_task
            y_opt.step()
            x_opt.step()

            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc=0
                    if F1_score_last>F1_score:
                        dc=1
                    F1_score_last=F1_score

 
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax={}'.format(x_itr,
                                                                                         100 * accuary(out,
                                                                                                       test.clean_target),
                                                                                         100 * p, 100 * r, 100 * F1_score,loss_F,pmax))

                    with open(log_path, 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([x_itr, 100. * acc, 100. * F1_score,total_time_iter,forward_time,backward_time,loss_F.item(),100*p,100*r,pmax])
        elif args.hg_mode=='BDAn_L1':
            F_list = []
            with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
                forward_time_task = time.time()
 

                for y_idx in range(args.y_loop):
                    out_f = fmodel(tr.data)

                    # print(loss_Lq(fmodel.parameters(),0.5,0.1))
                    loss_f = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                    total_loss_f = loss_f  # + args.y_L2_reg*loss_L2(grad_y)
                    diffopt.step(loss_f)
                    out_f = fmodel(tr.data)
                    # print(loss_Lq(fmodel.parameters(),0.5,0.1))
                    loss_f_new = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                    grad_y = torch.autograd.grad(loss_f_new, fmodel.parameters(), retain_graph=True)
                    F_list.append(
                        (F.cross_entropy(fmodel(val.data), val.clean_target) + args.y_L1_reg * loss_L1(grad_y)).item())

                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()

                pmax = F_list.index(max(F_list))

                #pmax_buffer.append(pmax)
                print(pmax)
                out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))

                #out_F = fmodel(val.data, params=fmodel.parameters())
                loss_F = F.cross_entropy(out_F, val.clean_target)
                out_f_new=fmodel(tr.data, params=fmodel.parameters(time=pmax + 1))
                loss_f_new = x(F.cross_entropy(out_f_new, tr.dirty_target, reduction='none'))
                grad_y=torch.autograd.grad(loss_f_new,fmodel.parameters(time=pmax + 1),retain_graph=True)
                total_loss_F = loss_F  + args.y_L1_reg*loss_L1(grad_y)
                grad_z = torch.autograd.grad(total_loss_F, fmodel.parameters(time=0), retain_graph=True)
                grad_x = torch.autograd.grad(total_loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
                    # print(p.data)
                for p, yp in zip(grad_z, y.parameters()):
                    if yp.grad == None:
                        yp.grad = p
                    else:
                        yp.grad += p

            backward_time_task = time.time() - backward_time_task
            backward_time += backward_time_task
            total_time_iter=time.time() - start_time_task
            y_opt.step()
            x_opt.step()

            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc=0
                    if F1_score_last>F1_score:
                        dc=1
                    F1_score_last=F1_score

 
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax={}'.format(x_itr,
                                                                                         100 * accuary(out,
                                                                                                       test.clean_target),
                                                                                         100 * p, 100 * r, 100 * F1_score,loss_F,pmax))

                    with open(log_path, 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([x_itr, 100. * acc, 100. * F1_score,total_time_iter,forward_time,backward_time,loss_F.item(),100*p,100*r,pmax])
        else:
            F_list = []
            with higher.innerloop_ctx(y, y_opt) as (fmodel, diffopt):
                forward_time_task = time.time()
  
                for y_idx in range(args.y_loop):
                    out_f = fmodel(tr.data)
                    # print(loss_Lq(fmodel.parameters(),0.5,0.1))
                    loss_f = x(F.cross_entropy(out_f, tr.dirty_target, reduction='none'))
                    diffopt.step(loss_f)
                    F_list.append(F.cross_entropy(fmodel(val.data), val.clean_target).item())
                pmax = F_list.index(max(F_list))
                #pmax_buffer.append(pmax)
                print(pmax)
                out_F = fmodel(val.data, params=fmodel.parameters(time=pmax + 1))
                forward_time_task = time.time() - forward_time_task
                forward_time += forward_time_task
                backward_time_task = time.time()
                #out_F = fmodel(val.data, params=fmodel.parameters())
                loss_F = F.cross_entropy(out_F, val.clean_target)
                #grad_z = torch.autograd.grad(loss_F, fmodel.parameters(time=0), retain_graph=True)
                grad_x = torch.autograd.grad(loss_F, x.parameters(), retain_graph=True)
                for p, xp in zip(grad_x, x.parameters()):
                    if xp.grad == None:
                        xp.grad = p
                    else:
                        xp.grad += p
 

            backward_time_task = time.time() - backward_time_task
            backward_time += backward_time_task
            total_time_iter=time.time() - start_time_task
            #y_opt.step()
            copy_parameter(y, fmodel,time=pmax+1)
            x_opt.step()

            if x_itr % 10 == 0:
                with torch.no_grad():
                    out = y(test.data)
                    acc = accuary(out, test.clean_target)
                    x_bi = Binarization(x.x.cpu().numpy())
                    clean = x_bi * tr.clean
                    acc_history.append(acc)
                    p = clean.mean() / (x_bi.sum() / x_bi.shape[0] + 1e-8)
                    r = clean.mean() / (1. - tr.rho)
                    F1_score = 2 * p * r / (p + r + 1e-8)
                    dc=0
                    if F1_score_last>F1_score:
                        dc=1
                    F1_score_last=F1_score

 
                    loss_F = F.cross_entropy(out, test.clean_target)
                    print('x_itr={},acc={:.3f},p={:.3f}.r={:.3f},F1 score={:.3f},val loss={:.3f},pmax={}'.format(x_itr,
                                                                                         100 * accuary(out,
                                                                                                       test.clean_target),
                                                                                         100 * p, 100 * r, 100 * F1_score,loss_F,pmax))

                    with open(log_path, 'a', encoding='utf-8', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([x_itr, 100. * acc, 100. * F1_score,total_time_iter,forward_time,backward_time,loss_F.item(),100*p,100*r,pmax])
 
