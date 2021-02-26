# Fed101
# Fed101@XLab.DaSE.ECNU

```shell script
cd Fed101/algorithm/FedAVG &
python fedavg-main.py
-dataset
femnist
-model
femnist
--lr
0.03
--lr-decay
0.99
--decay-step
1
--batch-size
10
--clients-per-round
10
--num-rounds
1000
--seed
12
--epoch
5
--eval-interval
1
--note
run_1_seed_12
```

```shell script
python fedavg-main.py 
-dataset femnist 
-model femnist 
--lr 0.03 
--lr-decay 0.99 
--batch-size 10 
--clients-per-round 10 
--num-rounds 1000 
--seed 24 
--epoch 5 
--eval-interval 1 
--note run_2_seed_24
```

```shell script
python fedavg-main.py -dataset femnist -model femnist --lr 0.03 --lr-decay 0.99 --batch-size 10 --clients-per-round 10 --num-rounds 1000 --seed 24 --epoch 5 --eval-interval 1 --note run_2_seed_24
```



## Dataset Overview

| dataset |  task  | metric | client | training set |   mean\|std\|skewness    | test set |   mean\|std\|skewness    |      partition       | link |
| :-----: | :----: | :----: | :----: | :----------: | :----------------------: | :------: | :----------------------: | :------------------: | :--: |
|  MNIST  | 10 clf |  acc   |  1000  |    61664     | 61.664\|144.63\|24751822 |   7371   | 7.371\|16.0772\|34058.3  |      power law       |      |
| FEMNIST | 62 clf |  acc   |  3400  |    671585    | 197.53\|76.681\|391488.3 |  77483   | 22.7891\|8.5105\|533.892 | realistic  partition |      |
| CIFAR10 | 10 clf |  acc   |  100   |    50000     |   500\|147.22\|-286980   |  10000   |        NA\|NA\|NA        |         LDA          |      |



## MNIST

### Description

​	1000 clients, refer to **fedprox**

### Model

​	CNN+FCNN

### Algorithm & Result

#### FedAVG

| #\T  |  50  |  60  |  65  |  70  |  75  |  80  | 85   |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | ---- |
|      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |

#### FedProx

| #\T  |  50  |  60  |  65  |  70  |  75  |  80  | 85   |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | ---- |
|      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |

#### FedSP

| #\T  |  50  |  60  |  65  |  70  |  75  |  80  | 85   |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | ---- |
|      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |

#### FedMC

| #\T  |  50  |  60  |  65  |  70  |  75  |  80  | 85   |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | ---- |
|      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |









## CIFAR10

### Description

​	100 clients (10 groups), for certain group, the clients belong to it share the 90% of the specified class.

### Model

​	CNN+FCNN

### Algorithm & Result

#### FedAVG

| #\T  |  30  |  35  |  40  |  45  |  50  |  55  | 60   | 61   | 62   | 63   | 64   | 65   | 66   | 67   | 68   | 69   | 70   | O\|R |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |

#### FedProx

```shell
u = 0.1
```

| #\T  |  30  |  35  |  40  |  45  |  50  |  55  | 60   | 61   | 62   | 63   | 64   | 65   | 66   | 67   | 68   | 69   | 70   | O\|R |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | :--: |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |

#### FedSP

| #\T  |  30  |  35  |  40  |  45  |  50  |  55  |  60  |  61  |  62  |  63  |  64  |  65  |  66  | 67   | 68   | 69   | 70   | O\|R |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | ---- | ---- | ---- | ---- | :--: |
|  1   |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |

#### FedMC

```shell
gradient penalty = 0
critic = 20
with sigmoid
```

| #\T  |  30  |  35  |  40  |  45  |  50  |  55  |  60  |  61  |  62  |  63  |  64  |  65  |  66  |  67  |  68  | 69   | 70   | O\|R |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | ---- | ---- | :--: |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |      |





## FEMNIST

### Overview

![Histogram](https://leaf.cmu.edu/webpage/images/femnist_hist.png)

### Description

​	Partition dataset based on the writer of the digit/character.

​	Sample clients based on the number of samples it has(>=350).

​	original dataset: 3500 clients and 785697 samples.

​	sampled subset: 503 clients, 193081 samples

### Model

​	CNN+FCNN

### Hyper-parameters

- **clients/round: 10/503**
- **epoch: 5**
- **batch-size: 10**
- **lr: 0.03**
- **lr-decay: 0.99**
- **decay-step: 1**
- **rounds: 1000**

### Algorithm & Result

#### FedAVG

| #\T  |    50    |    60     |    65     |    70     |    75     |    80     |     81     |     82     |     83     |     84     |  85  |    R\|O    |  Note   |
| :--: | :------: | :-------: | :-------: | :-------: | :-------: | :-------: | :--------: | :--------: | :--------: | :--------: | :--: | :--------: | :-----: |
|  1   | 9\|51.42 | 13\|60.43 | 17\|65.46 | 23\|70.41 | 38\|75.09 | 89\|80.21 | 104\|81.08 | 138\|82.26 | 204\|83.07 | 316\|84.08 |      | 600\|84.53 | seed=12 |
|      |          |           |           |           |           |           |            |            |            |            |      |            |         |
|      |          |           |           |           |           |           |            |            |            |            |      |            |         |
|      |          |           |           |           |           |           |            |            |            |            |      |            |         |

#### FedProx

```shell
u = 0.1
```

| #\T  |  50  |  60  |  65  |  70  |  75  |  80  |  81  |  82  |  83  |  84  |  85  | O\|R | Note |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  1   |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |

#### FedSP

| #\T  |  50  |  60  |  65  |  70  |  75  |  80  |  81  |  82  |  83  |  84  |  85  | O\|R | Note |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  1   |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |

#### FedMC

```shell
gradient penalty = 0
critic = 20
with sigmoid
```

| #\T  |  50  |  60  |  65  |  70  |  75  |  80  |  81  |  82  |  83  |  84  |  85  | O\|R | Note |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  1   |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |
|      |      |      |      |      |      |      |      |      |      |      |      |      |      |
