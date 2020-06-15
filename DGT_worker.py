from mpi4py import MPI
import numpy as np
from lr_model import lr_model
from data_generation import load_lr_data, generate_lr_data, standard_lr_solve

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#assign out/in neighbors
neighbors = 1
out_peers = [(rank + 1) % size, (rank + size - 1) % size]
in_peers = out_peers
in_deg = len(in_peers)
out_deg = len(out_peers)

#load lr data
lam = 1e-5
train_data, label, x_exact = load_lr_data('lr_data.mat', rank, size, lam)
[data_size, fea_size] = train_data.shape
lr_model = lr_model(train_data, data_size, fea_size, label, lam=lam)

#compute global objective for exact solution

opt_local_obj = lr_model.compute_obj(x_exact)
opt_global_obj = np.zeros(1)
comm.Barrier()
comm.Reduce(opt_local_obj, opt_global_obj, op=MPI.SUM, root=0)
opt_global_obj /= size


#initial state: x0, y0
local_x = np.random.normal(0, 4, (fea_size, ))
grad = lr_model.compute_grad(local_x)
last_grad = np.zeros((fea_size, ))
local_y = grad

iter_num = 300
lr = 8e-3

buf = np.empty(((fea_size) * 2 + 3) * out_deg, dtype=np.float)
MPI.Attach_buffer(buf)

sync = 1
for iter in range(iter_num):
    #send local x and y to out neighbors
    for i, outp in enumerate(out_peers):
        send_buff = np.stack((local_x - lr * local_y, local_y / (out_deg + 1)), axis=1)
        comm.Bsend([send_buff, MPI.F_FLOAT], dest=outp)

    #receive local x and y from in neighbors
    #clear receive buffer and flag
    x = np.zeros((fea_size, size))
    y = np.zeros((fea_size, size))
    recv_flag = np.zeros((size, ), dtype=int)

    if sync == 1:
        comm.Barrier()

    info = MPI.Status()
    while comm.Iprobe(source=MPI.ANY_SOURCE, status=info):
        recv_rank = info.source
        buffer = np.zeros((fea_size, 2))
        comm.Recv(buffer, source=recv_rank)
        x[:, recv_rank] += buffer[:, 0]
        y[:, recv_rank] += buffer[:, 1]
        recv_flag[recv_rank] += 1
        info = MPI.Status()
    in_deg = np.sum(recv_flag)
    #print(in_deg)

    #local update
    x[:, rank] = local_x - lr * local_y
    y[:, rank] = local_y / (out_deg + 1)

    #average sum to update local x
    local_x = np.sum(x, axis=1) / (in_deg + 1)# - lr * local_y

    #calculate gradient and update y
    last_grad = grad
    obj, grad = lr_model.compute_obj_grad(local_x)
    local_y = np.sum(y, axis=1) + grad - last_grad

    if sync == 1:
        #compute error
        aver_grad = np.zeros((fea_size, ))
        aver_x = np.zeros((fea_size, ))
        global_obj = np.zeros(1)
        cons_error = np.zeros(1)

        comm.Reduce(grad, aver_grad, op=MPI.SUM, root=0)
        aver_grad /= size

        comm.Allreduce(local_x, aver_x, op=MPI.SUM)
        aver_x /= size
        x_error = np.sqrt(np.dot(aver_x - x_exact, aver_x - x_exact))

        comm.Reduce(obj, global_obj, op=MPI.SUM, root=0)
        global_obj /= size

        comm.Reduce(np.dot(local_x - aver_x, local_x - aver_x), cons_error, root=0)
        cons_error = np.sqrt(cons_error)

        if rank == 0:
            print('iter: ', iter, 'consensus error', cons_error, 'global obj:', global_obj, 'exact obj', opt_global_obj, ', x error: ', x_error, ',log average gradient norm: ', np.log(np.sqrt(np.dot(aver_grad, aver_grad))))
    else:
        local_x_error = np.sqrt(np.dot(local_x - x_exact, local_x - x_exact))
        print('rank', rank, 'iter: ', iter, 'local x error', local_x_error)




