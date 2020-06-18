import matplotlib.pyplot as plt
import numpy as np
    
def interval_aver(train_log, aver_len):
    log_size = train_log.shape[0]
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
    return time_seq, loss_seq

if __name__ == '__main__':
    data_ind = 0
    asyn = 1
    asyn_train_log = np.load(str(1) + "DSGT_train_log_MNIST.npy")
    syn_train_log = np.load(str(0) + 'DSGT_train_log_MNIST.npy')
    print(asyn_train_log.shape, syn_train_log.shape)

    aver_len = 50
    time_seq, loss_seq = interval_aver(asyn_train_log, aver_len)
    print(len(time_seq), len(loss_seq))
    plt.plot(time_seq, loss_seq, color="b", linestyle="-", linewidth=1, label="async")
    
    time_seq, loss_seq = interval_aver(syn_train_log, aver_len)
    print(len(time_seq), len(loss_seq))
    plt.plot(time_seq, loss_seq, color="g", linestyle="-", linewidth=1, label="sync")

    plt.xlabel("time")
    plt.ylabel("loss")

    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

    plt.title("training loss")

    plt.savefig("DSGT_train_loss_MNIST.png")
    plt.show()
