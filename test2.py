#passRandomDraw.py
import numpy
from mpi4py import MPI
import time  # 引入time模块

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size= comm.Get_size()

dsize = 5
randNum = numpy.zeros([dsize])
randNum2 = numpy.zeros([dsize])

if rank == 0:
    randNum = numpy.random.random_sample(dsize)
    randNum2 = numpy.random.random_sample(dsize)
    sendbuf = numpy.stack((randNum, randNum2), axis=0)

    buf = numpy.empty(dsize + 10, dtype=numpy.float)
    MPI.Attach_buffer(buf)
    req = comm.Ibsend(sendbuf, dest=1)
    re = False
    i = 0
    while(re == False):
        re = MPI.Request.Test(req)
        i += 1
    print(rank, "send to", 1, 'count', i, 'time', time.time(), sendbuf)

if rank == 1:
    #comm.Ssend(randNum, dest=2)
    time.sleep(2)
    recbuf = numpy.zeros([2] + list(randNum.shape))
    req = comm.Irecv(recbuf, source=0)
    re = False
    i = 0
    while (re == False):
        re = MPI.Request.Test(req)
        i += 1
    print(rank, "received the number", 'count', i, 'time', time.time(), recbuf)
