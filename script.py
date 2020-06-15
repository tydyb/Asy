import torch
import torch.distributed as dist
import os

def run(rank, size):
    """ Distributed function to be implemented later. """
    pass
def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == '__main__':
    print(torch.distributed.is_available())
    init_process(0, 0, run, backend='gloo')



