# Fed101
# Fed101@XLab.DaSE.ECNU

## Experiments

| dataset       | accuracy | link |
| ------------- | -------- | ---- |
| FEMNIST       | 83.21%   |      |
| CIFAR10       | 83.00%   |      |
| synthetic_iid | 93.66%   |      |
|               |          |      |

- C: clients per communitions round
- E: epoch
- B: batch size
- T: threshold
- R: communication rounds

### MNIST

Fed 0

IID

Batch-size: 10

Epoch: 1

Clients Per Round: 10/100 (0.1)

实验结果：

| C             | E    | B    | T    | R    |
| ------------- | ---- | ---- | ---- | ---- |
| 10/ 100 (0.1) | 1    | 10   | 97%  |      |
|               |      |      |      |      |
|               |      |      |      |      |

