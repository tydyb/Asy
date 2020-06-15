import matplotlib.pyplot as plt
import numpy as np

data_ind = 0
train_log = np.load("ADSGT_train_log_MNIST.npy")
log_size = len(train_log)

train_log = np.array(train_log)
aver_len = 1
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

print(len(time_seq))
plt.plot(time_seq, loss_seq, color="k", linestyle="-", marker="o", linewidth=1, label="ADSGT")

plt.xlabel("time")
plt.ylabel("loss")

plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

plt.title("training loss")

plt.savefig("ADSGT_train_loss_MNIST.png")
plt.show()
