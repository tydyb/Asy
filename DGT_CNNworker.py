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
    output = model(data)#.to(device, non_blocking=True))
    loss = F.nll_loss(output, target)#.to(device, non_blocking=True))
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
    out_peers = [(rank + 1 + i) % size for i in range(3)]
    #[(rank + 1) % size, (rank + size - 1) % size]
    in_peers = [(rank - 1 - i + size) % size for i in range(3)]
    print(rank, out_peers, in_peers)
    in_deg = len(in_peers)
    out_deg = len(out_peers)

    #load data and model
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    gpu_ind = rank % torch.cuda.device_count()
    print(rank, "gpu", gpu_ind)
    device = torch.device("cuda:"+ str(gpu_ind) if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    #prepare training dataset
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
    local_dataset = torch.utils.data.Subset(global_dataset, ind)
   
    #move training data to gpu
    data_tensor = []
    target_tensor=[]
    for i in range(local_dataset.__len__()):
        data, target = local_dataset.__getitem__(i)
        data_tensor.append(data)
        target_tensor.append(torch.tensor(target))
    data_tensor = torch.stack(data_tensor).to(device)
    target_tensor = torch.stack(target_tensor).to(device)
    tensor_local_dataset = torch.utils.data.TensorDataset(data_tensor, target_tensor)
    
    train_loader = torch.utils.data.DataLoader(tensor_local_dataset, 
            batch_size=args.batch_size, shuffle=True)#, **kwargs)

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

    buf = np.empty(10 * out_deg * (net_size * 2 + 10), dtype=np.float)
    MPI.Attach_buffer(buf)

    #record if an neighbor has exited
    stop_flags = np.zeros([size, ], dtype=int)
    
    echo_interval = 100
    asyn = args.asyn
    send_req = []
    recv_deg = out_deg
    for i in range(iter_num):
        if not asyn:
            comm.Barrier()

        #send local x and y to out neighbors
        send_complete = MPI.Request.Testall(send_req)
        if recv_deg > 0 and send_complete == True:
            send_x = local_x - lr * local_y
            send_y = local_y / (out_deg + 1)
            send_num = np.minimum(recv_deg, out_deg)
            for outp in out_peers:#np.random.choice(out_peers, size=send_num, replace=False):
                send_buf = torch.stack((send_x, send_y), dim=0).cpu().numpy()
                tag = 0
                send_req.append(comm.Isend([send_buf, MPI.FLOAT], dest=outp, tag=tag))

        #receive local x and y from in neighbors
        #clear receive buffer and flag
        recv_flag = np.zeros((size, ), dtype=int)

        buf_x = torch.zeros(local_x.size(), requires_grad=False)
        buf_y = torch.zeros(local_y.size(), requires_grad=False)

        recv_deg = 0
        buf_size = [2, net_size]
        if asyn:
            waiting_time = 5
            s_time = time.time()
            while in_deg > 0 and recv_deg == 0:
                info = MPI.Status()
                while comm.Iprobe(source=MPI.ANY_SOURCE, status=info):
                    recv_rank = info.source
                    recv_tag = info.tag
                    #print(i, rank, "receiving", recv_rank)
                    recvbuf = np.zeros(buf_size, dtype=np.float32)
                    comm.Recv([recvbuf, MPI.FLOAT], source=recv_rank)
                    if recv_tag == 1:
                        stop_flags[recv_rank] = 1
                        if recv_rank in out_peers:
                            out_peers.remove(recv_rank)
                            out_deg -= 1
                        if recv_rank in in_peers:
                            in_peers.remove(recv_rank)
                            in_deg -= 1
                    else:
                        buf_x += torch.from_numpy(recvbuf[0, :])
                        buf_y += torch.from_numpy(recvbuf[1, :])
                        recv_flag[recv_rank] += 1
                        recv_deg += 1
                        acc_recv += 1
                    info = MPI.Status()
                    if recv_deg > in_deg:
                        break
                if time.time() - s_time > wait_time:
                    break
            wait_times -= 1
        else:
            for j, inp in enumerate(in_peers):
                recvbuf = np.zeros(buf_size, dtype=send_buf.dtype)
                comm.Recv([recvbuf, MPI.FLOAT], source=inp)
                buf_x += torch.from_numpy(recvbuf[0, :])
                buf_y += torch.from_numpy(recvbuf[1, :])
                recv_deg += 1    
                recv_flag[recv_rank] += 1
        buf_x = buf_x.to(device)
        buf_y = buf_y.to(device)
       
        if recv_deg > 0:
            # local update
            local_x -= lr * local_y
            local_y /= (out_deg + 1)
            
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
        if i % echo_interval == 0:
            print('rank', rank, 'iter: ', i, 'time', t_seq[i], 'local obj', obj)
        
    send_list = list(range(size))
    send_list.remove(rank)
    for node in send_list:
        sendbuf = np.zeros([1, ], )
        comm.Send(sendbuf, dest=node, tag=1)

    #receive the left iterates from neighbors
    '''
    info = MPI.Status()
    buffer = np.empty(buf_size, dtype=np.float32)
    while acc_recv < iter_num * in_deg:
        while comm.Iprobe(source=MPI.ANY_SOURCE, status=info):
            recv_rank = info.source
            comm.Recv([buffer, MPI.FLOAT], source=recv_rank)
            acc_recv += 1
            info = MPI.Status()
    '''
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
    np.save(str(int(args.asyn)) + 'DSGT_train_log_MNIST_n' + str(size - 1)  + '.npy', train_log)

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
    
    parser.add_argument('--asyn', type=bool, default=True, metavar='ASYN', help='asynchronous training or not')
    parser.add_argument('--iter-num', type=int, default=4000, metavar='ITER', help='iteration num per worker')
    args = parser.parse_args()

    if rank < size - 1:
        worker(new_comm, comm, args)
    else:
        monitor(comm, args)


