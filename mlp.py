import os
import secretflow as sf
from secretflow.data.vertical import read_csv as v_read_csv
from secretflow.preprocessing import LabelEncoder
from secretflow.preprocessing import OneHotEncoder
from secretflow.preprocessing import StandardScaler
from secretflow.data.split import train_test_split
from secretflow.stats.table_statistics import table_statistics
from secretflow.stats.ss_vif_v import VIF
from secretflow.stats.ss_pearsonr_v import PearsonR
import jax.numpy as jnp
import jax
import pandas as pd
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import sys
import subprocess
from data_process import spu, alice, bob, carol, data_dict

def install_package(package_name):
    '''
    安装指定的Python包
    '''
    try:
        # 使用 subprocess 执行 pip 安装命令，静默安装指定的包
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name, '-q'])
    except subprocess.CalledProcessError as e:
        # 安装出错则打印错误信息
        print(f"Error occurred while installing {package_name}: {e}")
    else:
        # 安装成功后打印成功信息
        print(f"Successfully installed {package_name}")
        return

# 安装0.6.0版本的flax包，用于构建神经网络
install_package('flax==0.6.0')

# 安装seaborn包，用于数据可视化
install_package('seaborn')

import seaborn as sns
from typing import Sequence
import flax.linen as nn
from sklearn.metrics import roc_auc_score

class MLP(nn.Module):  # 定义一个多层感知器（MLP）类
    features: Sequence[int]  # 每一层神经元的数量
    dropout_rate: float  # Dropout 概率
    @nn.compact
    def __call__(self, x, train, rngs=None):  # 定义前向传播过程
        for feat in self.features[:-1]:  # 遍历每一层（除了最后一层）
            x = nn.relu(nn.Dense(feat)(x))  # 应用全连接层和 ReLU 激活函数
            x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)  # 应用 Dropout
            x = nn.BatchNorm(use_running_average=not train, momentum=0.5, epsilon=1e-5)(x)  # 应用 Batch Normalization
        x = nn.Dense(self.features[-1])(x)  # 最后一层只应用全连接层
        return x  # 返回输出


def predict(params, x, train=False, rng_key=None):
    '''
    预测函数，传入权重偏置和输入，训练和测试都要用到
    '''
    from typing import Sequence
    import flax.linen as nn

    class MLP(nn.Module):  # 定义一个多层感知器（MLP）类
        features: Sequence[int]  # 每一层神经元的数量
        dropout_rate: float  # Dropout 概率
        @nn.compact
        def __call__(self, x, train, rngs=None):  # 定义前向传播过程
            for feat in self.features[:-1]:  # 遍历每一层（除了最后一层）
                x = nn.relu(nn.Dense(feat)(x))  # 应用全连接层和 ReLU 激活函数
                x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)  # 应用 Dropout
                x = nn.BatchNorm(use_running_average=not train, momentum=0.5, epsilon=1e-5)(x)  # 应用 Batch Normalization
            x = nn.Dense(self.features[-1])(x)  # 最后一层只应用全连接层
            return x  # 返回输出
    # FEATURE ：每一层神经元的数目
    FEATURES = [dim, 15, 8, 1]  # 定义每一层神经元的数量
    flax_nn = MLP(features=FEATURES, dropout_rate=0.1)  # 创建 MLP 实例，并设置 Dropout 概率为 0.1
    if rng_key is None:  # 如果没有提供随机数生成器的键
        rng_key = jax.random.PRNGKey(0)  # 使用默认的 PRNG 键
    y, updates = flax_nn.apply(  # 调用 MLP 的 apply 方法进行前向传播
        params,  # 模型参数
        x,  # 输入数据
        train,  # 是否处于训练模式
        mutable=['batch_stats'],  # 指示 Batch Normalization 的统计数据是可变的
        rngs={'dropout': rng_key}  # 传入用于 Dropout 的随机数生成器
    )
    batch_stats = updates['batch_stats']  # 获取 Batch Normalization 的统计数据
    return y  # 返回模型的预测结果

def loss_func(params, x, y, rng_key):
    '''
    使用MSE作为损失函数
    '''
    # 调用 predict 函数，使用给定的参数 params 和输入数据 x 进行预测，得到预测值 pred
    # 传递 train=True 和随机数生成键 rng_key 作为额外参数
    pred = predict(params, x, train=True, rng_key=rng_key)
    # 定义均方误差（MSE）函数
    def mse(y, pred):
        # 定义平方误差函数
        def squared_error(y, y_pred):
            # 计算每个样本的平方误差，并除以 2.0
            return jnp.multiply(y - y_pred, y - y_pred) / 2.0
        # 计算所有样本的平均平方误差
        return jnp.mean(squared_error(y, pred))
    # 调用 mse 函数，计算并返回 y 和 pred 之间的均方误差
    return mse(y, pred)

def train_auto_grad(X, y, params, batch_size=10, epochs=10, learning_rate=0.01):
    '''
    模型训练
    '''
    # 将输入数据 X 和标签 y 按照 batch_size 分割成多个小批次
    xs = jnp.array_split(X, len(X) // batch_size, axis=0)
    ys = jnp.array_split(y, len(y) // batch_size, axis=0)
    
    # 打印输入数据的shape
    #print(X.shape)
    
    # 初始化随机数生成器的键
    rng_key = jax.random.PRNGKey(0)
    
    # 进行多个 epoch 的训练
    for epoch in range(epochs):
        # 遍历每个小批次的数据
        for batch_x, batch_y in zip(xs, ys):
            # 计算当前批次的损失和梯度
            loss, grads = jax.value_and_grad(loss_func)(params, batch_x, batch_y, rng_key)
            
            # 使用梯度下降法更新模型参数
            params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    
    # 返回更新后的模型参数
    return params

class MLP_spu(nn.Module):  # 定义一个多层感知器（MLP）类 spu版本
    features: Sequence[int]  # 每一层神经元的数量
    @nn.compact
    def __call__(self, x):  # 定义前向传播过程
        for feat in self.features[:-1]:  # 遍历每一层（除了最后一层）
            x = nn.relu(nn.Dense(feat)(x))  # 应用全连接层和 ReLU 激活函数
        x = nn.Dense(self.features[-1])(x)  # 最后一层只应用全连接层
        return x  # 返回输出

def predict_spu(params, x):
    '''
    spu版本的预测函数
    '''
    # 从 typing 模块导入 Sequence 类型
    from typing import Sequence
    # 从 flax.linen 模块导入 nn
    import flax.linen as nn

    # 定义一个多层感知器（MLP）类，适用于 SPU
    class MLP_spu(nn.Module):
        # 定义每一层神经元的数量
        features: Sequence[int]
        
        # 定义前向传播过程
        @nn.compact
        def __call__(self, x):
            # 遍历每一层（除了最后一层）
            for feat in self.features[:-1]:  
                # 应用全连接层和 ReLU 激活函数
                x = nn.relu(nn.Dense(feat)(x))
            # 最后一层只应用全连接层
            x = nn.Dense(self.features[-1])(x)  
            # 返回输出
            return x

    # 定义每一层神经元的数量，FEATURES 列表
    FEATURES = [dim, 15, 8, 1]
    
    # 创建一个 MLP_spu 实例，传入 FEATURES 作为参数
    flax_nn = MLP_spu(features=FEATURES)
    
    # 使用给定的参数 params 和输入数据 x 进行预测，返回预测结果
    return flax_nn.apply(params, x)

def loss_func_spu(params, x, y):
    '''
    spu版本的损失函数
    '''
    # 使用给定的参数 params 和输入数据 x 进行预测，得到预测值 pred
    pred = predict_spu(params, x)
    # 定义均方误差（MSE）函数
    def mse(y, pred):
        # 定义平方误差函数
        def squared_error(y, y_pred):
            # 计算每个样本的平方误差，并除以 2.0
            return jnp.multiply(y - y_pred, y - y_pred) / 2.0
        
        # 计算所有样本的平均平方误差
        return jnp.mean(squared_error(y, pred))

    # 调用 mse 函数，计算并返回 y 和 pred 之间的均方误差
    return mse(y, pred)

def train_auto_grad_spu(X, y, params, batch_size=10, epochs=10, learning_rate=0.01):
    '''
    spu版本的模型训练函数
    '''
    # 将输入数据 X 按照 batch_size 分割成多个小批量，存储在 xs 列表中
    xs = jnp.array_split(X, len(X) // batch_size, axis=0)
    
    # 将目标数据 y 按照 batch_size 分割成多个小批量，存储在 ys 列表中
    ys = jnp.array_split(y, len(y) // batch_size, axis=0)
    
    # 打印输入数据 X 的形状
    # print(X.shape)
    
    # 进行 epochs 次训练迭代
    for epoch in range(epochs):
        # 遍历每个小批量数据
        for batch_x, batch_y in zip(xs, ys):
            # 计算当前批量数据的损失值和梯度
            loss, grads = jax.value_and_grad(loss_func_spu)(params, batch_x, batch_y)
            
            # 更新模型参数，使用梯度下降法
            params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    
    # 返回训练后的模型参数
    return params

def cpu_version_mlp(X_train_plaintext, y_train_plaintext, init_params, batch_size, epochs, learning_rate):
    '''
    在 CPU 上训练和评估MLP模型
    '''
    # 使用自动微分方法训练模型，返回训练后的参数
    params = train_auto_grad(
        X_train_plaintext, y_train_plaintext, init_params, batch_size, epochs, learning_rate
    )
    
    # 设置用于预测的随机数生成器的键
    rng_key = jax.random.PRNGKey(1)
    
    # 使用训练后的参数进行预测
    y_pred = predict(params, X_test_plaintext, train=False)
    
    # 计算并打印模型的 AUC 分数
    os.system('clear')
    print(f"\033[31m(Flax NN CPU) auc: {roc_auc_score(y_test_plaintext, y_pred)}\033[0m")

def spu_version_mlp(X_train_spu, y_train_spu, params_spu, batch_size, epochs, learning_rate):
    '''
    在 SPU 上训练和评估MLP模型
    '''
    # 使用 SPU 环境中的 train_auto_grad 函数训练模型，返回训练后的参数
    params_spu = spu(
        train_auto_grad_spu, static_argnames=['batch_size', 'epochs', 'learning_rate']
    )(
        X_train_spu,  # 训练数据
        y_train_spu,  # 训练标签
        params_spu,  # 初始参数
        batch_size=batch_size,  # 批次大小
        epochs=epochs,  # 训练轮数
        learning_rate=learning_rate  # 学习率
    )
    
    # 使用 SPU 环境中的 predict 函数进行预测
    y_pred_spu = spu(predict_spu)(params_spu, X_test_spu)
    
    # 将预测结果从 SPU 环境中揭示出来
    y_pred_ = sf.reveal(y_pred_spu)
    
    # 计算并打印模型的 AUC 分数
    print(f"\033[31m(Flax NN SPU) auc: {roc_auc_score(y_test_plaintext, y_pred_)}\033[0m")

if __name__ == '__main__':
    # 获取数据集
    X_train_spu = data_dict['X_train_spu']
    y_train_spu = data_dict['y_train_spu']
    X_test_spu = data_dict['X_test_spu']
    y_test_spu = data_dict['y_test_spu']

    # 将 SPU 上的训练和测试数据揭露为明文数据
    X_train_plaintext = sf.reveal(X_train_spu)  # 揭露训练集特征数据
    y_train_plaintext = sf.reveal(y_train_spu)  # 揭露训练集目标数据
    X_test_plaintext = sf.reveal(X_test_spu)  # 揭露测试集特征数据
    y_test_plaintext = sf.reveal(y_test_spu)  # 揭露测试集目标数据

    # 获取训练集特征数据的维度
    dim = X_train_plaintext.shape[1]

    # 定义每一层神经元的数量
    FEATURES = [dim, 15, 8, 1]

    # 创建 CPU 版本的多层感知器（MLP）实例，设置 Dropout 概率为 0.1
    flax_nn = MLP(features=FEATURES, dropout_rate=0.1)

    # 创建 SPU 版本的多层感知器（MLP）实例
    flax_nn_spu = MLP_spu(features=FEATURES)

    # 根据数据集的特征维度设置特征维度
    feature_dim = dim

    if len(sys.argv[1])==3:
        # 设置模型训练的参数
        epochs = sys.argv[1][0]  # CPU 版本的训练轮数
        learning_rate = sys.argv[1][1]  # CPU 版本的学习率
        batch_size = sys.argv[1][2] # CPU 版本的批量大小
        epochs_spu = sys.argv[1][0]  # SPU 版本的训练轮数
        learning_rate_spu   = sys.argv[1][1]  # SPU 版本的学习率
        batch_size_spu = sys.argv[1][2]  # SPU 版本的批量大小
    else:
        epochs = 2  # CPU 版本的训练轮数
        learning_rate = 0.02  # CPU 版本的学习率
        batch_size = 100 # CPU 版本的批量大小
        epochs_spu = 2  # SPU 版本的训练轮数
        learning_rate_spu = 0.02  # SPU 版本的学习率
        batch_size_spu = 100 # SPU 版本的批量大小
    # 初始化 CPU 版本的模型参数，使用随机数生成键和全 1 的输入数据
    init_params = flax_nn.init(jax.random.PRNGKey(1), jnp.ones((batch_size, feature_dim)), train=False)

    # 初始化 SPU 版本的模型参数，使用随机数生成键和全 1 的输入数据
    init_params_spu = flax_nn_spu.init(jax.random.PRNGKey(1), jnp.ones((batch_size, feature_dim)))

    # 将初始化的 SPU 版本模型参数从 Alice 传递到 SPU
    params = sf.to(alice, init_params_spu).to(spu)

    # 打印训练集特征数据的形状
    # print(X_train_plaintext.shape)

    # 调用 CPU 版本的 MLP 训练函数，传入训练集特征数据、目标数据、初始化参数、批量大小、训练轮数和学习率
    cpu_version_mlp(X_train_plaintext, y_train_plaintext, init_params, batch_size, epochs, learning_rate)

    # 调用 SPU 版本的 MLP 训练函数，传入训练集特征数据、目标数据、初始化参数、批量大小、训练轮数和学习率
    spu_version_mlp(X_train_spu, y_train_spu, params, batch_size_spu, epochs_spu, learning_rate_spu)