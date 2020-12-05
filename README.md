# Fed101
# Fed101@XLab.DaSE.ECNU

```sh
cd Fed101 &
python main.py
-dataset
mnist
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

### MNIST

- overview: 60000 training examples and 10000 test examples.

#### IID

- partitioning: Data is shuffled  and then partitioned into 100 clients each receiving 600 examples.

|  Model   | Clients Per Round | Epoch | Batchsize | Threshold/Rounds | Optimal/Rounds |
| :------: | :---------------: | :---: | :-------: | :--------------: | :------------: |
| MNIST2NN |   10/ 100 (0.1)   |   1   |    10     |     97%/144      |   98.09%/630   |
|          |                   |       |           |                  |                |
|          |                   |       |           |                  |                |

#### Non-IID

