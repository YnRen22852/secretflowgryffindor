# 导入必要的Python库
import os  # os模块提供与操作系统交互的功能
import secretflow as sf  # 引入secretflow库，支持联邦学习与隐私保护计算
from secretflow.data.vertical import read_csv as v_read_csv  # 从secretflow中导入用于垂直拆分数据的read_csv函数
from secretflow.preprocessing import LabelEncoder  # 导入标签编码器
from secretflow.preprocessing import OneHotEncoder  # 导入独热编码器
from secretflow.preprocessing import StandardScaler  # 导入标准化数据的标准化器
from secretflow.data.split import train_test_split  # 从secretflow导入数据拆分函数
from secretflow.stats.table_statistics import table_statistics  # 从secretflow导入统计分析函数
from secretflow.stats.ss_vif_v import VIF  # 导入VIF（方差膨胀因子）模块
from secretflow.stats.ss_pearsonr_v import PearsonR  # 导入皮尔逊相关系数模块
import jax.numpy as jnp  # 导入JAX中的numpy，用于高效的数值计算
import jax  # JAX库，支持自动微分和加速运算
import pandas as pd  # 用于数据处理的Pandas库
import numpy as np  # 用于数组和矩阵计算的Numpy库
import tempfile  # 用于临时文件的创建
import os  # 用于操作系统相关功能的os库
import matplotlib.pyplot as plt  # 用于绘图的Matplotlib库
import sys  # 用于与Python解释器进行交互的sys库
import subprocess  # 用于执行外部命令的subprocess库
from data_process import spu, alice, bob, carol, data_dict  # 从data_process中导入预定义的数据处理和数据字典

def install_package(package_name):
    '''
    安装指定的Python包
    '''
    try:
        # 使用 subprocess 执行 pip 安装命令，静默安装指定的包
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name, '-q'])
    except subprocess.CalledProcessError as e:
        # 如果安装过程出错，捕获并处理异常
        # print(f"Error occurred while installing {package_name}: {e}")
    else:
        # 如果安装成功，输出成功信息
        # print(f"Successfully installed {package_name}")
        return

# 安装flax==0.6.0版本的包，用于构建神经网络
install_package('flax==0.6.0')

# 安装seaborn包，用于数据可视化（目前注释掉，未使用）
# install_package('seaborn')

# 导入seaborn，用于数据可视化
import seaborn as sns
# 从typing模块导入Sequence，用于定义列表或元组类型
from typing import Sequence
# 导入flax.linen模块，用于构建神经网络
import flax.linen as nn
# 从sklearn导入roc_auc_score，用于计算AUC分数
from sklearn.metrics import roc_auc_score

# 定义多层感知器（MLP）类，继承自flax.linen.Module
class MLP(nn.Module):
    features: Sequence[int]  # 每一层神经元的数量
    dropout_rate: float  # Dropout层的丢弃概率
    
    @nn.compact
    def __call__(self, x, train, rngs=None):
        '''
        定义前向传播过程
        '''
        # 遍历每一层（除了最后一层）
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))  # 使用全连接层和ReLU激活函数
            x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)  # 使用Dropout层
            x = nn.BatchNorm(use_running_average=not train, momentum=0.5, epsilon=1e-5)(x)  # 使用Batch Normalization层
        # 最后一层只应用全连接层
        x = nn.Dense(self.features[-1])(x)
        return x  # 返回模型的输出

# 预测函数，传入权重偏置和输入，训练和测试都要用到
def predict(params, x, train=False, rng_key=None):
    '''
    用于模型预测
    '''
    from typing import Sequence
    import flax.linen as nn

    # 定义多层感知器（MLP）类
    class MLP(nn.Module):
        features: Sequence[int]  # 每一层神经元的数量
        dropout_rate: float  # Dropout概率
        
        @nn.compact
        def __call__(self, x, train, rngs=None):
            for feat in self.features[:-1]:
                x = nn.relu(nn.Dense(feat)(x))
                x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
                x = nn.BatchNorm(use_running_average=not train, momentum=0.5, epsilon=1e-5)(x)
            x = nn.Dense(self.features[-1])(x)
            return x

    # 设置每一层神经元的数量
    FEATURES = [dim, 15, 8, 1]  # 定义每层神经元的数量
    flax_nn = MLP(features=FEATURES, dropout_rate=0.1)  # 创建MLP实例，设置Dropout概率为0.1

    # 如果未提供随机数生成器的键，使用默认值
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)  # 使用默认的PRNG密钥

    # 应用MLP进行前向传播
    y, updates = flax_nn.apply(
        params,  # 模型参数
        x,  # 输入数据
        train,  # 是否处于训练模式
        mutable=['batch_stats'],  # Batch Normalization使用可变的统计数据
        rngs={'dropout': rng_key}  # 用于Dropout的随机数生成器
    )
    batch_stats = updates['batch_stats']  # 获取Batch Normalization的统计数据
    return y  # 返回模型的预测结果

# 定义均方误差（MSE）作为损失函数
def loss_func(params, x, y, rng_key):
    '''
    使用MSE作为损失函数
    '''
    # 通过预测函数获得预测值
    pred = predict(params, x, train=True, rng_key=rng_key)

    # 定义MSE函数
    def mse(y, pred):
        # 计算平方误差
        def squared_error(y, y_pred):
            return jnp.multiply(y - y_pred, y - y_pred) / 2.0
        
        # 返回平均平方误差
        return jnp.mean(squared_error(y, pred))

    # 调用mse函数，计算并返回损失
    return mse(y, pred)

# 定义训练函数，使用自动微分更新模型参数
def train_auto_grad(X, y, params, batch_size=10, epochs=10, learning_rate=0.01):
    '''
    模型训练过程
    '''
    # 将数据分批次进行训练
    xs = jnp.array_split(X, len(X) // batch_size, axis=0)
    ys = jnp.array_split(y, len(y) // batch_size, axis=0)

    # 初始化随机数生成器的键
    rng_key = jax.random.PRNGKey(0)

    # 进行多个epoch的训练
    for epoch in range(epochs):
        # 遍历每个批次数据
        for batch_x, batch_y in zip(xs, ys):
            # 计算当前批次的损失和梯度
            loss, grads = jax.value_and_grad(loss_func)(params, batch_x, batch_y, rng_key)

            # 使用梯度下降法更新模型参数
            params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)

    # 返回更新后的模型参数
    return params

# 定义用于SPU版本的多层感知器（MLP）类
class MLP_spu(nn.Module):
    features: Sequence[int]  # 每层神经元的数量
    
    @nn.compact
    def __call__(self, x):
        '''
        定义前向传播过程
        '''
        # 遍历每一层（除了最后一层）
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))  # 使用全连接层和ReLU激活函数
        x = nn.Dense(self.features[-1])(x)  # 最后一层只应用全连接层
        return x  # 返回输出

# 定义SPU版本的预测函数
def predict_spu(params, x):
    '''
    spu版本的预测函数
    '''
    from typing import Sequence
    import flax.linen as nn

    # 定义多层感知器（MLP）类
    class MLP_spu(nn.Module):
        features: Sequence[int]  # 每层神经元的数量

        @nn.compact
        def __call__(self, x):
            # 遍历每一层（除了最后一层）
            for feat in self.features[:-1]:
                x = nn.relu(nn.Dense(feat)(x))
            # 最后一层只应用全连接层
            x = nn.Dense(self.features[-1])(x)
            return x

    # 设置每一层神经元的数量
    FEATURES = [dim, 15, 8, 1]

    # 创建SPU版本的MLP模型
    flax_nn = MLP_spu(features=FEATURES)
    
    # 使用给定参数和输入进行预测
    return flax_nn.apply(params, x)

# 定义SPU版本的损失函数
def loss_func_spu(params, x, y):
    '''
    spu版本的损失函数
    '''
    # 使用SPU版本的预测函数进行预测
    pred = predict_spu(params, x)

    # 定义MSE函数
    def mse(y, pred):
        def squared_error(y, y_pred):
            return jnp.multiply(y - y_pred, y - y_pred) / 2.0
        
        return jnp.mean(squared_error(y, pred))

    # 调用MSE函数，计算损失并返回
    return mse(y, pred)

# 定义SPU版本的模型训练过程
def train_auto_grad_spu(X, y, params, batch_size=10, epochs=10, learning_rate=0.01):
    '''
    spu版本的模型训练函数
    '''
    # 将输入数据按batch_size进行分割
    xs = jnp.array_split(X, len(X) // batch_size, axis=0)
    
    # 将目标数据y按batch_size进行分割
    ys = jnp.array_split(y, len(y) // batch_size, axis=0)

    # 进行epochs次训练
    for epoch in range(epochs):
        # 遍历每个批次数据
        for batch_x, batch_y in zip(xs, ys):
            # 计算损失和梯度
            loss, grads = jax.value_and_grad(loss_func_spu)(params, batch_x, batch_y)

            # 使用梯度下降法更新模型参数
            params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)

    # 返回训练后的模型参数
    return params

# 定义CPU版本的MLP训练与评估过程
def cpu_version_mlp(X_train_plaintext, y_train_plaintext, init_params, batch_size, epochs, learning_rate):
    '''
    在CPU上训练和评估MLP模型
    '''
    # 使用自动微分训练模型，返回训练后的参数
    params = train_auto_grad(
        X_train_plaintext, y_train_plaintext, init_params, batch_size, epochs, learning_rate
    )
    
    # 设置随机数生成器的键
    rng_key = jax.random.PRNGKey(1)
    
    # 使用训练后的模型参数进行预测
    y_pred = predict(params, X_test_plaintext, train=False)
    
    # 计算并输出AUC分数
    os.system('clear')  # 清屏操作
    print(f"\033[31m(Flax NN CPU) auc: {roc_auc_score(y_test_plaintext, y_pred)}\033[0m")

# 定义SPU版本的MLP训练与评估过程
def spu_version_mlp(X_train_spu, y_train_spu, params_spu, batch_size, epochs, learning_rate):
    '''
    在SPU上训练和评估MLP模型
    '''
    # 使用SPU环境训练模型
    params_spu = spu(
        train_auto_grad_spu, static_argnames=['batch_size', 'epochs', 'learning_rate']
    )(
        X_train_spu,  # 训练数据
        y_train_spu,  # 训练标签
        params_spu,  # 初始参数
        batch_size=batch_size,  # 批量大小
        epochs=epochs,  # 训练轮数
        learning_rate=learning_rate  # 学习率
    )
    
    # 使用SPU环境的预测函数进行预测
    y_pred_spu = spu(predict_spu)(params_spu, X_test_spu)
    
    # 揭示SPU上的预测结果
    y_pred_ = sf.reveal(y_pred_spu)
    
    # 计算并输出AUC分数
    print(f"\033[31m(Flax NN SPU) auc: {roc_auc_score(y_test_plaintext, y_pred_)}\033[0m")

# 主函数，初始化数据并调用训练和评估过程
if __name__ == '__main__':
    # 获取训练和测试数据
    X_train_spu = data_dict['X_train_spu']
    y_train_spu = data_dict['y_train_spu']
    X_test_spu = data_dict['X_test_spu']
    y_test_spu = data_dict['y_test_spu']

    # 揭示训练和测试数据为明文数据
    X_train_plaintext = sf.reveal(X_train_spu)  # 揭示训练集特征
    y_train_plaintext = sf.reveal(y_train_spu)  # 揭示训练集目标
    X_test_plaintext = sf.reveal(X_test_spu)  # 揭示测试集特征
    y_test_plaintext = sf.reveal(y_test_spu)  # 揭示测试集目标

    # 获取特征维度
    dim = X_train_plaintext.shape[1]

    # 定义每一层的神经元数量
    FEATURES = [dim, 15, 8, 1]

    # 创建CPU和SPU版本的多层感知器模型
    flax_nn = MLP(features=FEATURES, dropout_rate=0.1)
    flax_nn_spu = MLP_spu(features=FEATURES)

    # 根据数据集维度确定特征维度
    feature_dim = dim

    # 根据命令行参数设置训练参数
    if len(sys.argv[1]) == 3:
        epochs = sys.argv[1][0]  # CPU训练轮数
        learning_rate = sys.argv[1][1]  # CPU学习率
        batch_size = sys.argv[1][2]  # CPU批次大小
        epochs_spu = sys.argv[1][0]  # SPU训练轮数
        learning_rate_spu = sys.argv[1][1]  # SPU学习率
        batch_size_spu = sys.argv[1][2]  # SPU批次大小
    else:
        epochs = 2  # CPU训练轮数
        learning_rate = 0.02  # CPU学习率
        batch_size = 100  # CPU批次大小
        epochs_spu = 2  # SPU训练轮数
        learning_rate_spu = 0.02  # SPU学习率
        batch_size_spu = 100  # SPU批次大小

    # 初始化CPU版本的模型参数
    init_params = flax_nn.init(jax.random.PRNGKey(1), jnp.ones((batch_size, feature_dim)), train=False)
    
    # 初始化SPU版本的模型参数
    init_params_spu = flax_nn_spu.init(jax.random.PRNGKey(1), jnp.ones((batch_size, feature_dim)))

    # 将初始化的SPU模型参数从Alice传递到SPU
    params = sf.to(alice, init_params_spu).to(spu)

    # 调用CPU版本的MLP训练函数
    cpu_version_mlp(X_train_plaintext, y_train_plaintext, init_params, batch_size, epochs, learning_rate)

    # 调用SPU版本的MLP训练函数
    spu_version_mlp(X_train_spu, y_train_spu, params, batch_size_spu, epochs_spu, learning_rate_spu)
