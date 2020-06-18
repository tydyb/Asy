from mpi4py import MPI
import numpy as np
from LENET import Model
import torch
from torchvision import datasets, transforms
import torch.optim as optim
import argparse
import torch.nn.functional as F
import time

def worker_grad(model, device, data, target, optimizer):
    optimizer.zero_grad()
    output = model(data.to(device))
    loss = F.nll_loss(output, target.to(device))
    loss.backward()

    return loss.item()

def convert_net_to_sendbuf(Net, sendbuf):
    pass

def assign_array_to_net(sendbuf, Net):
    pos = 0
    for j, param in enumerate(Net.parameters()):
        param.data.copy_(sendbuf[pos: pos + param.view(-1).size()[0]].reshape(param.shape))
        pos += param.view(-1).size()[0]

def worker(comm, whole_comm, args):
    #distributed setting
    rank = comm.Get_rank()
    size = comm.Get_size()

    #assign out/in neighbors
    neighbors = 1
    out_peers = [(rank + 1) % size, (rank + size - 1) % size]
    print(out_peers)
    in_peers = out_peers
    in_deg = len(in_peers)
    out_deg = len(out_peers)

    #load data and model
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    global_dataset = datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    data_size = len(global_dataset)
    seg = np.floor(data_size / size)
    rmd = data_size % size
    if rank + 1 <= rmd:
        seg += 1
        local_a = rank * seg
    else:
        local_a = rank * seg + rmd
    ind = np.arange(local_a, local_a + seg, dtype=np.int)
    train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(global_dataset, ind),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    #sample a batch
    trainloader_iter = iter(train_loader)
    try:
        batch, target = trainloader_iter.next()
    except StopIteration:
        trainloader_iter = iter(train_loader)
        batch, target = trainloader_iter.next()

    #start time
    start_t = time.time()

    #initial state: x0, y0
    Net = Model().to(device)
    Net.train()
    local_x = []
    for param in Net.parameters():
        local_x.append(param.data.view(-1).clone())
    local_x = torch.cat(local_x).detach()
    net_size = local_x.size()[0]

    optimizer = optim.SGD(Net.parameters(), lr=args.lr)
    obj = worker_grad(Net, device, batch, target, optimizer)
    grad = []
    for param in Net.parameters():
        grad.append(param.grad.view(-1).clone())
    grad = torch.cat(grad).detach()
    local_y = grad.clone().detach()
    last_grad = torch.zeros_like(local_y, device=device, requires_grad=False)

    iter_num = args.iter_num
    lr = args.lr
    decay = 1

    #time series
    t_seq = np.zeros([iter_num, ], dtype=np.float32)
    loss_seq = np.zeros([iter_num, ], dtype=np.float32)

    #the number of accumulated received iterates
    acc_recv = 0

    buf = np.empty(4 * out_deg * (net_size * 2 + 10))
    MPI.Attach_buffer(buf)

    asyn = args.asyn
    for i in range(iter_num):
        # local update
        local_x -= lr * local_y
        #optimizer.step()
        local_y /= (out_deg + 1)

        #send local x and y to out neighbors
        send_buf = torch.stack((local_x, local_y), dim=0).cpu().numpy()
        buf_size = send_buf.shape
        for j, outp in enumerate(out_peers):
            #print(i, rank, 'send to', outp)
            comm.Bsend([send_buf, MPI.FLOAT], dest=outp)

        #receive local x and y from in neighbors
        #clear receive buffer and flag
        recv_flag = np.zeros((size, ), dtype=int)

        if ~ asyn:
            comm.Barrier()

        buf_x = torch.zeros(local_x.size(), requires_grad=False)
        buf_y = torch.zeros(local_y.size(), requires_grad=False)

        info = MPI.Status()
        recv_deg = 0
        while (acc_recv < iter_num * in_deg) and (recv_deg < 1 * in_deg):
            while comm.Iprobe(source=MPI.ANY_SOURCE, status=info):
                recv_rank = info.source
                buffer = np.empty(buf_size, dtype=send_buf.dtype)
                comm.Recv([buffer, MPI.FLOAT], source=recv_rank)
                #print(i, rank, 'receive from',  recv_rank)
                #buffer = torch.from_numpy(buffer)
                buf_x += torch.from_numpy(buffer[0, :])
                buf_y += torch.from_numpy(buffer[1, :])
                recv_flag[recv_rank] += 1
                recv_deg += 1
                acc_recv += 1
                info = MPI.Status()
                if recv_deg > 2.5 * in_deg:
                    break
        
        buf_x = buf_x.to(device)
        buf_y = buf_y.to(device)
        
        #average consensus and update local x
        buf_x += local_x
        local_x = buf_x / (recv_deg + 1)
        buf_y += local_y

        #update net parameters
        assign_array_to_net(local_x, Net)

        #compute gradient and update localy
        last_grad.copy_(grad)
        try:
            batch, target = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(train_loader)
            batch, target = next(trainloader_iter)
        obj = worker_grad(Net, device, batch, target, optimizer)
        grad = []
        for param in Net.parameters():
            grad.append(param.grad.view(-1).clone())
        grad = torch.cat(grad).detach()
        local_y = buf_y + grad - last_grad
        lr *= decay

        t_seq[i] = time.time() - start_t
        loss_seq[i] = obj
        if ~ asyn:
            #compute error
            '''
            global_obj = np.zeros(1)
            cons_error = np.zeros(1)
            aver_x = np.zeros(list(local_x.size()), dtype=np.float32)

            comm.Allreduce(local_x.numpy(), aver_x, op=MPI.SUM)
            aver_x /= size

            comm.Reduce(np.array(obj), global_obj, op=MPI.SUM, root=0)
            global_obj /= size

            local_error = local_x.numpy() - aver_x
            comm.Reduce(np.dot(local_error, local_error), cons_error, root=0)
            cons_error = np.sqrt(cons_error)
            '''
            #if rank == 0:
            print('rank', rank, 'iter: ', i, 'time', t_seq[i], 'local obj', obj) #, 'consensus error', cons_error, 'global obj:', global_obj)
        else:
            #whole_comm.isend([t_seq[i], obj], dest=whole_comm.Get_size() - 1)
            print('rank', rank, 'iter: ', i, 'time', t_seq[i], 'local obj', obj)

    #receive the left iterates from neighbors
    info = MPI.Status()
    buffer = np.empty(buf_size, dtype=np.float32)
    while acc_recv < iter_num * in_deg:
        while comm.Iprobe(source=MPI.ANY_SOURCE, status=info):
            recv_rank = info.source
            comm.Recv([buffer, MPI.FLOAT], source=recv_rank)
            print("after training, ", rank, "receive from", recv_rank)
            acc_recv += 1
            info = MPI.Status()

    whole_comm.send(np.stack((t_seq, loss_seq), axis=1).tolist(), dest=whole_comm.Get_size() - 1)
    whole_comm.Send([local_x.cpu().numpy(), MPI.FLOAT], dest=whole_comm.Get_size() - 1)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def monitor(comm, args):
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(size)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    #average model
    Net = Model().to(device)
    net_size = sum(x.numel() for x in Net.parameters())
    aver_x = np.zeros([net_size, ], dtype=np.float)

    #list of asynchronous training loss
    loss_rec = []

    #the number of workers that have stopped training
    stop_num = 0
    stop_flag = np.zeros([size - 1, ])

    while stop_num < size - 1:
        info = MPI.Status()
        while comm.Iprobe(source=MPI.ANY_SOURCE, status=info):
            recv_rank = info.source
            if stop_flag[recv_rank] == 1:
                buffer = np.zeros([net_size, ], dtype=np.float32)
                print('monitor receives model from', recv_rank)
                comm.Recv([buffer, MPI.FLOAT], source=recv_rank)
                aver_x += buffer
                stop_num += 1
            else:
                buffer = comm.recv(source=recv_rank)
                stop_flag[recv_rank] = 1
                loss_rec.extend(buffer)
            info = MPI.Status()

    #save train log
    train_log = np.array(loss_rec)
    log_size = train_log.shape[0]
    print(train_log.shape)
    ind = np.argsort(train_log[:, 0], axis=0)
    train_log = train_log[ind, :]

    aver_len = size - 1
    loss_seq = []
    time_seq = []
    i = 0
    while i < log_size:
        time_seq.append(train_log[i, 0])
        end = i + aver_len
        if end > log_size:
            end = log_size
        loss_seq.append(np.average(train_log[i:end, 1]))
        i += aver_len
    train_log = np.stack((np.array(time_seq), np.array(loss_seq)), axis=1)
    np.save(str(args.asyn) + 'DSGT_train_log_MNIST.npy', train_log)

    aver_x = torch.from_numpy(aver_x / (size - 1))
    assign_array_to_net(aver_x, Net)

    #test model
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True, **kwargs)
    test(Net, device, test_loader)

    if args.save_model:
        torch.save(Net.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank < size - 1:
        color = 0
    else:
        color = 1
    new_comm = MPI.COMM_WORLD.Split(color=color)

    # training setting
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    parser.add_argument('--asyn', type=int, default=1, metavar='ASYN', help='asynchronous training or not')
    parser.add_argument('--iter-num', type=int, default=4000, metavar='ITER', help='iteration num per worker')
    args = parser.parse_args()

    if rank < size - 1:
        worker(new_comm, comm, args)
    else:
        monitor(comm, args)


