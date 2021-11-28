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

def _diag_add(mat_in, diag_elem, inplace=False):
    mat_out = mat_in
    if not inplace:
        mat_out = mat_out.clone()
    mat_out.diagonal().add_(diag_elem)
    return mat_out

class o_NGPlus(Optimizer):

    def __init__(self, params, lr=1e-1, alpha=0.99, momentum=0.9, damping=0.001, weight_decay=0, update_freq=1, epsilon=1e-8,cov_update_freq=10):
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                weight_decay=weight_decay, update_freq=update_freq,cov_update_freq=cov_update_freq, alpha=alpha, epsilon=epsilon)
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

                if len(state) == 0:
                    state['step'] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone().detach()
                    dim = grad.size(0)
                    state['MatFisher'] = epsilon * torch.eye(dim,out=grad.new(dim,dim))
                    state['InvMatFisher'] = torch.eye(dim, out=grad.new(dim, dim))

                if weight_decay > 0:
                    grad = grad.add(p, alpha=weight_decay)

                if momentum > 0:
                    state['momentum_buffer'].mul_(momentum).add_(grad)
                    grad = state['momentum_buffer']
                
                dw = grad.view(grad.size(0),-1)
                if state['step'] % group['cov_update_freq'] == 0:
                    state['MatFisher'].addmm_(mat1=dw, mat2=dw.t(), beta=alpha, alpha=(1.0-alpha))

                if state['step'] % group['update_freq'] == 0:                   
                    MatFisher = state['MatFisher']
                    # corr_inv = 1 / (1 - alpha ** (state['step']/2+1)) 
                    state['InvMatFisher'] = _diag_add(MatFisher, (min(damping,max(torch.max(torch.abs(MatFisher)),1e-3) ))**0.5).inverse().contiguous()

                                   
                InvMatFisher = state['InvMatFisher']
                dw = torch.mm(InvMatFisher,dw)
                grad = dw.contiguous().view(*g_shape)
                state['step'] += 1
                p.add_(grad, alpha=-group['lr'])

        return loss
