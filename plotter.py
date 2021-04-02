import numpy as np
import csv
import matplotlib.pyplot as plt

num_epochs = 5
fig = plt.figure(figsize=(14, 7))

for phase in ['train', 'valid']:
    epoch = []
    loss = []
    acc =[]
    ix = 0
    for i in range(num_epochs):
        with open('./logs/{}-log-epoch-{:02d}.txt'.format(phase, i+1), 'r') as f: # 1--> i+1
            # df = csv.reader(f, delimiter='\t')
            df = f.readline()
            df = df.split(" ")
            while '' in df:
                df.remove('')
            data = list(df)
        epoch.append(float(data[0]))
        loss.append(float(data[1]))
        acc.append(float(data[3]))
        ix += 1
    print("epoch:", epoch)
    print("loss:", loss)
    print("acc:", acc)
    plt.subplot(1, 2, 1)
    if phase == 'train':
        plt.plot(epoch, loss, label=phase, color='red', linewidth=3.0)
    else:
        plt.plot(epoch, loss, label=phase, color='blue', linewidth=3.0)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)

    plt.subplot(1, 2, 2)
    plt.tight_layout()

    if phase == 'train':
        train_acc = plt.plot(epoch, acc, label=phase, color='red', linewidth=3.0)
    else:
        val_acc = plt.plot(epoch, acc, label=phase, color='blue', linewidth=3.0)

    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)

    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 20})

plt.savefig('./result/VGG19_BN.png', dpi=fig.dpi)
# plt.show()
