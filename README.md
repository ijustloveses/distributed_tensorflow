
environment
=============
- server 1: dual gpus (gtx1080) with tensorflow-gpu version 1.2.1
- server 2: single gpu (gtx1080) with tensorflow-gpu version 1.2.1

项目采用 MNIST 数据，运行一个比较简单的两层 full connected neural network，共计运行 100 个 EPOCH

每个 BATCH 100 个训练样本，每个 EPOCH 运行 6 个 BATCH

tfsingle.py
=============
普通的单机 tf 程序，运行在 server 1 上，并指定使用 /gpu:0

每个 epoch 1.3 秒左右，运行完毕之后，准确率为 72%

BETWEEN GRAPH: tfdist_between.py
==================================

该结构中包含的参数均通过 ps 作业(/job:ps)进行声明并使用tf.train.replica_device_setter() 方法将参数 round robin 映射到不同的 ps 任务中

tf.train.replica_device_setter 中的 worker_device 参数指定后续 op 操作所在的 worker device

### 1. 单机运行，采用 ps + worker 架构，1 ps + 1 worker; 异步训练，不过由于只有一个 worker，同步异步都一样

```
settings.py

ps_svrs = ['localhost:2223']
worker_svrs = ['localhost:2222']
```

```
nohup python tfdist_between.py --job_name=ps --task_index=0 > ps.log 2>&1 &
nohup python tfdist_between.py --job_name=worker --task_index=0 > worker.log 2>&1 &
```

运行在 server1 上，每个 epoch 3.8 秒左右，运行完毕之后，准确率同样为 72%

看到，由于 ps 和 worker 的通信，导致每个 epoch 显著的降低了时间

### 2. 单机运行，采用 ps + worker 架构，1 ps + 2 workers; 异步训练

```
ps_svrs = ['localhost:2223']
worker_svrs = ['localhost:2222', 'localhost:2221']
```

为了避免 ps 和 worker1 把显存占光，那么需要修改代码

```
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index, config=config)
```

```
nohup python tfdist_between.py --job_name=ps --task_index=0 > ps.log 2>&1 &
nohup python tfdist_between.py --job_name=worker --task_index=0 > worker.log 2>&1 &
nohup python tfdist_between.py --job_name=worker --task_index=1 > worker1.log 2>&1 &
```

运行在 server1 上

- worker0 会先运行，它是 chief worker，每个 Epoch 运行 5.45 秒，比上面一节的 3.8 秒要更慢一些，训练中使用 /job:worker/replica:0/task:0/gpu:0
- worker0 运行完 100 个 EPOCH 之后，准确率为 0.8，看到比单机要高，这是因为它训练中还异步的同步了 worker1 的结果
- worker1 比 worker0 晚一些运行，每个 Epoch 也是运行 5.45 秒左右，和 worker0 差不多，训练中使用 /job:worker/replica:0/task:1/gpu:0
- worker1 运行完 100 个 EPOCH 之后，准确率也为 0.8，看到比单机要高，这是因为它训练中还异步的同步了 worker0 的结果

看到，这里采用的是异步训练的方式，故此 worker0 和 worker1 并没有在每个 batch 结束后同步参数再进行下一 batch 的训练，而是各自训练，异步更新

两个 workers 一共进行了 200 个 EPOCH 的训练，故此结果要比单机的 100 次 (0.8 > 0.72)，但是由于是异步更新，应该不如单机进行 200 次训练好

由于参与通信的节点多了一个，故此比起上面一节，每个 Epoch 训练的时间又长了一些  ...

gpu-utils

```
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 00000000:01:00.0 Off |                  N/A |
| 35%   53C    P2    40W / 180W |    653MiB /  8114MiB |     17%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 1080    Off  | 00000000:02:00.0 Off |                  N/A |
| 27%   34C    P8     5W / 180W |    355MiB /  8114MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### 3. 同上，但是使用 gpu1

为了让 worker1 使用 gpu1，修改代码如下：

```
-        worker_device="/job:worker/task:%d" % FLAGS.task_index,
+        worker_device="/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, FLAGS.task_index),
```

结果 worker0 从 5.45 秒降低到 5.24 秒左右；worker1 从 5.45 秒降低到 5.35 秒左右；

看到，确实使用多 gpu 能提高一些效率，但是对比 grpc 通信带来的代价，提升并不明显；

可以想象，如果这里换成比较复杂的模型，那么多 gpu 带来的提升会更高一些！然而本模型过于简单，多 gpu 提升不明显

gpu-utils

```
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   48C    P2    38W / 180W |    521MiB /  8114MiB |      8%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce GTX 1080    Off  | 00000000:02:00.0 Off |                  N/A |
| 28%   44C    P2    40W / 180W |    487MiB /  8114MiB |      3%      Default |
+-------------------------------+----------------------+----------------------+
```


