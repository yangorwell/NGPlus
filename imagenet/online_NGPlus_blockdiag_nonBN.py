# An implementation of NG+ optimizer from:
#
# NG+ : A Multi-Step Matrix-Product Natural Gradient Method for Deep Learning
# Minghan Yang, Dong Xu, Qiwen Cui, Zaiwen Wen, Pengxiang Xu
# Preprint Paper: https://arxiv.org/abs/2106.07454
# contact: yangminghan at pku.edu.cn, taroxd at pku.edu.cn, wenzw at pku.edu.cn

# Copyright (c) 2021 Minghan Yang, Dong Xu, Qiwen Cui, Zaiwen Wen, Pengxiang Xu
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that
# the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
#    following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#    and the following disclaimer in the documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.RWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch
from torch.optim.optimizer import Optimizer
from .param_utils import get_norm_parameters, get_common_parameters, get_norm_bias_parameters
def compute_block_diagonal(mat):
    row = mat.size()[0]
    b_row = row % 128
    # print(row,'sddddddd')
    if b_row !=0 and row!= b_row:
        a_row = row - b_row
        a , b = torch.split(mat,a_row,0)
        c = torch.matmul(b,b.t())
        a = a.view(-1,128,mat.size()[1])
        d = torch.bmm(a,a.transpose(1,2))
        tmp = torch.block_diag(*d,c)
    elif b_row ==0:
        a = mat.view(-1,128,mat.size()[1])
        d = torch.bmm(a,a.transpose(1,2))
        tmp = torch.block_diag(*d)
    else:
        tmp = torch.mm(mat, mat.t())
    return tmp


def block_diag_update_Precond(state,dw,block_size,row_last,dim):
    partition_num = dim//block_size
    for i in range(partition_num):
        dw[i*block_size:(i+1)*block_size,:] = torch.mm(state['InvMatFisher'][0][i],dw[i*block_size:(i+1)*block_size,:])
    if row_last > 0:
        dw[partition_num*block_size:,:] = torch.mm(state['InvMatFisher'][1],dw[partition_num*block_size:,:])  
    return dw

def block_diag_update_InvMatFisher(state,damping, block_size,row_last,dim):
    partition_num = dim//block_size
    for i in range(partition_num):
        state['MatFisher'][0][i] = _diag_add(state['MatFisher'][0][i],damping)
    state['InvMatFisher'][0] = state['MatFisher'][0].inverse().contiguous()
    if row_last > 0:
        state['InvMatFisher'][1] = _diag_add(state['MatFisher'][1],damping).inverse().contiguous()


def block_diag_update_MatFisher(state,dw,block_size,row_last,dim,alpha=0.9):
    partition_num = dim//block_size
    for i in range(partition_num):
        tmp = dw[i*block_size:(i+1)*block_size,:]
        state['MatFisher'][0][i].addmm_(mat1=tmp, mat2=tmp.t(), beta=alpha, alpha=(1.0-alpha))
    if row_last > 0:
        tmp = dw[partition_num*block_size:,:]
        # print(tmp.size())
        state['MatFisher'][1].addmm_(mat1=tmp, mat2=tmp.t(), beta=alpha, alpha=(1.0-alpha))


def _diag_add(mat_in, diag_elem, inplace=False):
    mat_out = mat_in
    if not inplace:
        mat_out = mat_out.clone()
    mat_out.diagonal().add_(diag_elem)
    return mat_out

class o_NGPlus(Optimizer):

    def __init__(self, params, lr=1e-1, alpha=0.99, momentum=0.9, damping=0.001, weight_decay=0, update_freq=1, epsilon=1e-8,cov_update_freq=100,block_diag=True):
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                weight_decay=weight_decay, update_freq=update_freq,cov_update_freq=cov_update_freq, alpha=alpha, epsilon=epsilon,block_diag=block_diag)
        super(o_NGPlus, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                g_shape = grad.shape
                original_size = grad.size()
                state = self.state[p]
                alpha = group['alpha']
                momentum = group['momentum']
                damping = group['damping']
                weight_decay = group['weight_decay']
                epsilon = group['epsilon']
                block_diag = group['block_diag']


                block_size = 1024
                dim = grad.size(0)
                row_last = dim%block_size
                partition_num = dim // block_size
                block_flag = block_diag and dim > block_size

                if len(state) == 0:
                    state['step'] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone().detach()
                    
                    if weight_decay >= 0:
                        if block_flag:
                            state['MatFisher'] = [torch.empty([partition_num,block_size,block_size])]
                            for block in range(partition_num):
                                state['MatFisher'][0][block] = epsilon * torch.eye(block_size)
                            if row_last > 0:
                                state['MatFisher'].append(epsilon*torch.eye(row_last))
                            state['InvMatFisher'] = state['MatFisher'].copy()
                        else:
                            state['MatFisher'] = epsilon * torch.eye(dim,out=grad.new(dim,dim))
                            state['InvMatFisher'] = torch.eye(dim, out=grad.new(dim, dim))

                if weight_decay > 0:
                    grad = grad.add(p, alpha=weight_decay)

                if momentum > 0:
                    state['momentum_buffer'].mul_(momentum).add_(grad)
                    grad = state['momentum_buffer']
                
                dw = grad.view(grad.size(0),-1)
                if state['step'] % group['cov_update_freq'] == 0 and weight_decay >= 0:
                    if block_flag:
                        block_diag_update_MatFisher(state,dw,alpha,block_size,row_last,dim)
                        # state['MatFisher'].addmm_(mat1=dw, mat2=dw.t(), beta=alpha, alpha=(1.0-alpha))
                    else:
                        # state['MatFisher'].mul_(alpha).add_(compute_block_diagonal(dw),alpha=(1.0-alpha))
                        state['MatFisher'].addmm_(mat1=dw, mat2=dw.t(), beta=alpha, alpha=(1.0-alpha))

                if state['step'] % group['update_freq'] == 0 and weight_decay >= 0:                   
                    # corr_inv = 1 / (1 - alpha ** (state['step']/2+1)) 
                    if block_flag:
                        block_diag_update_InvMatFisher(state,damping,block_size,row_last,dim)
                    else:
                        MatFisher = state['MatFisher']
                        state['InvMatFisher'] = _diag_add(MatFisher, (min(damping,max(torch.max(torch.abs(MatFisher)),1e-3) ))**0.5).inverse().contiguous()

                if weight_decay >= 0:   
                    if block_flag: 
                        dw = block_diag_update_Precond(state,dw,block_size,row_last,dim)
                    else:
                        InvMatFisher = state['InvMatFisher']
                    # print(state['step'],InvMatFisher.size())
                        dw = torch.mm(InvMatFisher,dw)

                    grad = dw.contiguous().view(*g_shape)
                state['step'] += 1
                p.add_(grad, alpha=-group['lr'])

        return loss


def create_oNG_optimizer(model, lr,weight_decay=0, alpha=0.99, momentum=0.9,
    damping=0.001, update_freq=1, epsilon=1e-8,cov_update_freq=100,
    block_diag=True,exclude_layers=['bn', 'ln', 'bias'], bias_correction=False):
    # can only exclude BatchNorm, LayerNorm, bias layers
    # ['bn', 'ln'] will exclude BatchNorm, LayerNorm layers
    # ['bn', 'ln', 'bias'] will exclude BatchNorm, LayerNorm, bias layers
    # [] will not exclude any layers
    if 'bias' in exclude_layers:
        params = [
            dict(params=get_common_parameters(
                model, exclude_func=get_norm_bias_parameters)),
            dict(params=get_norm_bias_parameters(model), weight_decay=0)
        ]
    elif len(exclude_layers) > 0:
        params = [
            dict(params=get_common_parameters(
                model, exclude_func=get_norm_parameters)),
            dict(params=get_norm_parameters(model), weight_decay=0)
        ]
    else:
        params = model.parameters()
    optimizer = o_NGPlus(params, lr,  weight_decay=weight_decay,alpha=alpha,
        momentum=momentum,damping=damping,update_freq=update_freq,epsilon=epsilon,
        cov_update_freq=cov_update_freq,block_diag=block_diag)
    return optimizer
