import numpy
from mpi4py import MPI
import time  # 引入time模块

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size= comm.Get_size()

dsize = 10000
randNum = numpy.random.random_sample(dsize)
recvbuf1 = numpy.zeros(dsize)
recvbuf2 = numpy.zeros(dsize)

request = []
request.append(comm.Isend(randNum, dest=(rank + 1) % size))
request.append(comm.Isend(randNum, dest=(rank + 2) % size))
print(rank, "send", randNum)
request.append(comm.Irecv(recvbuf1, source=(rank + size - 1) % size))
request.append(comm.Irecv(recvbuf2, source=(rank + size - 2) % size))
MPI.Request.Waitall(request)
print(rank, "receive", recvbuf1, recvbuf2)

