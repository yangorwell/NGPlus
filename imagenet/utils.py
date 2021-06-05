import os
import shutil
import argparse
import torch

class MPIEnv:
    def __init__(self, world_size=None, rank=None, local_rank=None):
        if world_size is None:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
        assert world_size > 0
        if rank is None:
            rank = int(os.environ.get('RANK', 0))
        assert (rank>=0) and (rank<world_size)
        if local_rank is None:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        assert (local_rank>=0) and (local_rank<world_size)
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.is_master = (rank==0)
        self.is_distributed = (world_size > 1)
        self.is_local_rank0 = (local_rank==0)

    def __str__(self):
        ret = f'MPIEnv(world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank})'
        return ret

    __repr__ = __str__


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--datadir', default='/mnt/ILSVRC2012', help='Place where data are stored')
    parser.add_argument('--logdir', default='log', type=str, help='where logs go')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight_decay')
    parser.add_argument('--lr-decay-rate', default=0.87, type=float)
    parser.add_argument('--label-smoothing', default=0.1, type=float, help='label smoothing parameter')
    parser.add_argument('--damping', default=0.05, type=float, help='damping')
    parser.add_argument('--cov-update-freq', default=500, type=int, help='The frequency to update fisher matrix.')
    parser.add_argument('--inv-update-freq', default=500, type=int, help='The frequency to update inverse fisher matrix')
    parser.add_argument('--cov-running-average', action='store_true', help='Computes running average for fisher matrix.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--local_rank', default=0, type=int, help='provided by torch.distributed.launch')
    parser.add_argument('--short-epoch', action='store_true', help='make epochs short (for debugging)')
    parser.add_argument('--sua', action='store_true', help='Use sua approximation')
    parser.add_argument('--print_freq', default=500, type=int, metavar='N', help='log/print every this many steps')
    parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode')
    parser.add_argument('--warmup_epoch', default=0, type=int, help='first k epoch to gradually increase learning rate')
    parser.add_argument('--lr_warmup', default=0.01, type=float, help='warmup learning rate')
    parser.add_argument('--epoch_end', default=60, type=int)
    parser.add_argument('--lr_exponent', default=6, type=float)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--method',default='exponent',type=str,choices=['exponent','poly','linear'])
    parser.add_argument('--lr_init', default=0.1, type=float, help='learning rate')
    parser.add_argument('--curvature_momentum', default=0.1, type=float, help='curvature_matrix_momentum')
    parser.add_argument('--batch_size', default=64, type=int, help='the number of data samples per GPUs')
    parser.add_argument('--decay_epochs', default=100, type=int, help='epoch for decay learning rate')
    return parser



def check_phase(lr_phase, dl_phase):
    assert all(('lr' in x) and ('bs' not in x) for x in lr_phase)
    assert all(('lr' not in x) and ('bs' in x) for x in dl_phase)

    assert all(len(x['lr'])==2 for x in lr_phase)
    assert all(len(x['ep'])==2 for x in lr_phase), 'linear learning rates must contain end epoch'
    assert all(x['ep'][0]<x['ep'][1] for x in lr_phase)
    assert lr_phase[0]['ep'][0]==0
    assert all(x['ep'][1]==y['ep'][0] for x,y in zip(lr_phase[:-1],lr_phase[1:]))
    for x in lr_phase:
        if 'type' not in lr_phase:
            x['type'] = 'linear'
    assert {x['type'] for x in lr_phase} <= {'linear','exp'}

    assert dl_phase[0]['ep']==0
    assert len({x['ep'] for x in dl_phase})==len(dl_phase)
    for x in dl_phase:
        assert 'val_bs' in x
        if 'rect_val' not in x:
            x['rect_val'] = False
        if 'min_scale' not in x:
            x['min_scale'] = 0.08
    return lr_phase, dl_phase


def save_file_for_reproduce(save_list, logdir):
    assert os.path.abspath(save_list[0])==save_list[0], 'first element should be __file__'
    dirname = os.path.dirname(save_list[0])
    save_list = [save_list[0]] + [os.path.join(dirname,x) for x in save_list[1:]]
    for x in save_list:
        if os.path.exists(x):
            shutil.copy2(x, logdir)
        else:
            print(f'WARNING: file "{x}" not exists')


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=1)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing/(pred.shape[1]-1))
            true_dist.scatter_(1, target.view(-1,1), 1-self.smoothing)
        return torch.mean(torch.sum(-true_dist*pred, dim=1))


class LRScheduler:
    def __init__(self, optimizer, phase):
        self.optimizer = optimizer
        self.epoch_to_phase = {y:x for x in phase for y in range(*x['ep'])}
        self.lr = None
        self.lr_epoch_start = None
        self.lr_epoch_end = None

    def update_lr(self, epoch, ind_batch, batch_tot, warmup_epoch,lr_warmup,lr_init,method,epoch_end,lr_exponent,lr_decay_rate,decay_epoch):
        phase = self.epoch_to_phase[epoch]
        # lr0, lr1 = phase['lr']
        ep0, ep1 = phase['ep']
        # assert phase['type'] in {'linear','exp'}
        # if phase['type']=='linear':
            # self.lr = lr0 + (lr1-lr0) * tmp0
        if epoch < warmup_epoch:
            tmp0 = (epoch  + ind_batch/batch_tot) / (warmup_epoch)
            self.lr = lr_warmup + (lr_init-lr_warmup) * tmp0
        else: #exp
            # self.lr = lr0*(lr1/lr0)**tmp0
            if method =='poly':
                self.lr = lr_init * (1- (epoch -warmup_epoch + ind_batch/batch_tot)/epoch_end)**lr_exponent
            else:
                self.lr = lr_init * (lr_decay_rate**(epoch  + ind_batch/batch_tot - warmup_epoch))
            if epoch > decay_epoch:
                self.lr  = self.lr/5

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        if epoch < warmup_epoch:
            self.lr_epoch_start = lr_warmup + (lr_init - lr_warmup)*(epoch - ep0) / (ep1 - ep0)
            self.lr_epoch_end = lr_warmup + (lr_init - lr_warmup)*(epoch + 1 - ep0) / (ep1 - ep0)
        else:
            if method =='poly':
                self.lr_epoch_start = lr_init * (1- (epoch + 1 - ep1 )/epoch_end)**lr_exponent
                self.lr_epoch_end   = lr_init * (1- (epoch - ep1 + 2)/epoch_end)**lr_exponent
            else:
                self.lr_epoch_start = lr_init * (lr_decay_rate**(epoch   - warmup_epoch))
                self.lr_epoch_end   = lr_init * (lr_decay_rate**(epoch  + 1 - warmup_epoch))


def my_topk(logits, label, topk=(1,)):
    prediction = logits.topk(max(topk), dim=1)[1]
    tmp0 = prediction==label.view(-1,1)
    ret = [tmp0[:,:x].sum().item() for x in topk]
    return ret


def sum_tensor(tensor):
    ret = tensor.clone()
    torch.distributed.all_reduce(ret, op=torch.distributed.ReduceOp.SUM)
    return ret
