# Fed101
# Fed101@XLab.DaSE.ECNU

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

