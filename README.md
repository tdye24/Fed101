# Fed101
# Fed101@XLab.DaSE.ECNU

```sh
cd Fed101 &
python main.py
-dataset
mnist
-partitioning
iid
-model
mnist2nn
-model-path
./mnist/mnist.pkl
--lr
3e-4
--batch-size
10
--clients-per-round
10
--num-rounds
3000
--seed
24
--epoch
1
--eval-interval
1
```

### MNIS                     T

- overview: 60000 training examples and 10000 test examples.

  - |      label       |  0   |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |  9   |
    | :--------------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
    | training samples | 5923 | 6742 | 5958 | 6131 | 5842 | 5421 | 5918 | 6265 | 5851 | 5949 |
    |   test samples   | 980  | 1135 | 1032 | 1010 | 982  | 892  | 958  | 1028 | 974  | 1009 |

#### IID

- partitioning: Data is shuffled  and then partitioned into 100 clients each receiving 600 examples.

|  Model   | Parameters | Clients Per Round | Epoch | Batchsize | Threshold\|Rounds | Optimal\|Rounds |
| :------: | :--------: | :---------------: | :---: | :-------: | :---------------: | :-------------: |
| MNIST2NN |   199210   |   10/ 100 (0.1)   |   1   |    10     |    97.02%\|144    |   98.09%\|630   |
| MNIST2NN |   199210   |   10/ 100 (0.1)   |   1   | $\infty$  |   97.02%\|2025    |  #97.53%\|3000  |
| MNISTCNN |   582026   |   10/ 100 (0.1)   |   5   |    10     |    99.01%\|93     |   99.23%\|984   |

Note that # means that it does not fully converge until the predefined communitions rounds, 3000 here.

#### Non-IID

|  Model   | Parameters | Clients Per Round | Epoch | Batchsize | Threshold\|Rounds | Optimal\|Rounds |
| :------: | :--------: | :---------------: | :---: | :-------: | :---------------: | :-------------: |
| MNIST2NN |   199210   |   10/ 100 (0.1)   |   1   |    10     |         N         |  *90.70%\|1009  |
| MNIST2NN |   199210   |   10/ 100 (0.1)   |   1   | $\infty$  |         N         |  #51.72%\|2914  |
| MNIST2NN |   199210   |   100/ 100(1.0)   |   1   | $\infty$  |         N         |  #49.99%\|2993  |
| MNISTCNN |   582026   |   10/ 100 (0.1)   |   5   |    10     |         N         |  98.05%\|1074   |
| MNISTCNN |   582026   |   100/ 100(1.0)   |   5   |    10     |         N         |  *96.90%\|817   |
| MNISTCNN |   582026   |   100/ 100(1.0)   |   5   | $\infty$  |                   |                 |

Note that * means that it does not converge and oscillates quite a bit.