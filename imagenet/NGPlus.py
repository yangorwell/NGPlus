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
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn.functional as F

def _diag_add(mat_in, diag_elem, inplace=False):
    mat_out = mat_in
    if not inplace:
        mat_out = mat_out.clone()
    mat_out.diagonal().add_(diag_elem)
    return mat_out


def _inv_covs( ggt, damping):
    iggt = _diag_add(ggt, (damping )**0.5).inverse().contiguous()
    return  iggt


def _precond(weight, bias, group, state):
    iggt = state['iggt']
    g = weight.grad.data
    s = g.shape
    if group['layer_type'] == 'Conv2dEx':
        g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
    if bias is not None:
        gb = bias.grad.data
        g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
    g = torch.mm(iggt,g)
    if bias is not None:
        gb = g[:, -1].contiguous().view(*bias.shape)
        g = g[:, :-1]
    else:
        gb = None
    g = g.contiguous().view(*s)
    return g, gb


def _compute_covs(group, state, x, gy, alpha):
    mod = group['mod']

    shape_x = x.shape
    shape_gy = gy.shape
    batch_size = shape_gy[0]
    if group['layer_type'] == 'Conv2dEx':
        factor = 1 / (batch_size * (shape_gy[2]*shape_gy[3])**0.5)
        gy = gy.view(gy.size(0),gy.size(1),-1)
    else:
        factor = 1/batch_size
        gy = gy.unsqueeze(2)
    
    if group['layer_type'] == 'Conv2dEx':
        x = F.unfold(x, mod.kernel_size, padding=mod.padding, stride=mod.stride)
    else:
        x = x.unsqueeze(2)
    if mod.bias is not None:
        ones = torch.ones_like(x[:, :1])
        x = torch.cat([x, ones], dim=1)
    dw = torch.bmm(gy,x.transpose(1,2))
    dw = dw.detach().permute(1, 0, 2)
    dw = dw.contiguous().view(dw.size(0), -1)
    if 'ggt' not in state:
        state['ggt'] = torch.mm(dw, dw.t()) * factor
    else:
        state['ggt'].addmm_(mat1=dw, mat2=dw.t(), beta=(1-alpha), alpha=alpha*factor)
    del dw



class NGPlus(torch.optim.Optimizer):

    def __init__(self, net, damping, update_freq=1, alpha=1.0, loss_scaler=None, world_size=1):
        """ NG+ Preconditionner for Linear and Conv2d layers.
        Args:
            net (torch.nn.Module): Network to precondition.
            damping (float): regularization parameter for the inverses.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).            
        """
        self.fp16 = loss_scaler is not None
        self.loss_scaler = loss_scaler
        self.world_size = world_size
        self.damping = damping
        self.update_freq = update_freq
        self.alpha = alpha
        self.params = []
        self.iteration_counter = 0
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['LinearEx', 'Conv2dEx']:
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                self.params.append({'params':params, 'mod':mod, 'layer_type':mod_class})
        super(NGPlus, self).__init__(self.params, {})

    def compute_all_covs(self):
        async_handles = []
        for group in self.param_groups:
            mod = group['mod']
            state = self.state[group['params'][0]]
            x = mod.last_input
            # loss function is averaged per batch, so multiply it with batch size per process
            gy = mod.last_output.grad
            if self.fp16:
                assert (x.dtype==torch.float16) and (gy.dtype==torch.float16)
                x = x.float()
                gy = gy.float() / self.loss_scaler.get_scale()
            gy = gy * mod.last_output.grad.shape[0]
            _compute_covs(group, state, x, gy, self.alpha)
            if self.world_size>1:
                def all_avg(key):
                    state[key] = state[key].contiguous() / self.world_size
                    handle = torch.distributed.all_reduce(state[key], torch.distributed.ReduceOp.SUM, async_op=True)
                    async_handles.append(handle)
                all_avg('ggt')
        for handle in async_handles:
            handle.wait()


    def step(self):
        for i, group in enumerate(self.param_groups):
            state = self.state[group['params'][0]]
            if self.iteration_counter % self.update_freq == 0:
                iggt = _inv_covs(state['ggt'], self.damping)
                state['iggt'] = iggt.contiguous()
        for group in self.param_groups:
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            gw, gb = _precond(weight, bias, group, state)
            # Updating gradients
            weight.grad.data = gw
            if bias is not None:
                bias.grad.data = gb
        
        self.iteration_counter += 1
