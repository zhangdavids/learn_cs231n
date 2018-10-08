


1. ckpt 文件  文件名字写错了。。


检查点文件 checkpoint 文件 扩展名字一般为 .ckpt

图协议文件 二进制文件 扩展为 .pb
用 tf.train.write_graph()保存， 只包含图形结构，然后使用tf.import_graph_def()
来加载图形


2. 使用卷积神经网络 训练次数 目前是5000次  为达到好的效果 最好20000以上

使用 gpu 更好一些


mnist 的几种解法比较

