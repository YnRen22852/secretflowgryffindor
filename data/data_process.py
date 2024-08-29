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
import pandas as pd
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns

def download_dataset(full_file_path):
    '''
    将数据集下载到本地
    '''
    # UCI Adult 数据集的 URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    # 列名
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]

    # 下载数据集
    adult_df = pd.read_csv(url, names=column_names, sep=',\s', na_values=["?"], engine='python')
    # 保存为csv文件
    adult_df.to_csv(full_file_path,index=False)

def load_dataset(full_file_path):
    '''
    在指定目录家在数据集，如果不存在则下载
    '''
    # 检查文件路径是否存在，如果不存在则下载数据集
    if not os.path.exists(full_file_path):
        download_dataset(full_file_path)
    
    try:
        # 尝试读取 CSV 文件到 DataFrame
        df = pd.read_csv(full_file_path)
    except Exception as e:
        # 如果读取文件时发生错误，抛出带有错误信息的 ValueError
        raise ValueError(f"文件读取错误:{e}")
    
    # 添加一列 'uid'，其值为 DataFrame 的索引加 1
    df['uid'] = df.index + 1
    
    # 返回处理后的 DataFrame
    return df

def split_dataset(data_df, alice_path, bob_path, carol_path):
    '''
    分割数据集，将数据集垂直切分，并且存储在在各个参与方的临时文件路径
    '''
    # 获取数据集的列数，减去 1 是因为最后一列是 'uid'
    num_columns = data_df.shape[1] - 1
    
    # 设置分割点，将列数分成三部分
    split_point1 = num_columns // 3
    split_point2 = 2 * num_columns // 3
    
    # 分割数据集，并且确保每一方都拥有 'uid' 列
    # Alice 获取前 split_point1 列和最后一列 'uid'，并随机抽取 90% 的数据
    data_alice = data_df.iloc[:, np.r_[0:split_point1, -1]].sample(frac=0.9)
    
    # Bob 获取从 split_point1 到 split_point2 列和最后一列 'uid'，并随机抽取 90% 的数据
    data_bob = data_df.iloc[:, np.r_[split_point1:split_point2, -1]].sample(frac=0.9)
    
    # Carol 获取从 split_point2 到最后一列（包括 'uid'），并随机抽取 90% 的数据
    data_carol = data_df.iloc[:, split_point2:].sample(frac=0.9)
    
    # 将三方数据集保存至 CSV 文件
    data_alice.reset_index(drop=True).to_csv(alice_path, index=False)
    data_bob.reset_index(drop=True).to_csv(bob_path, index=False)
    data_carol.reset_index(drop=True).to_csv(carol_path, index=False)
    
    # 返回保存的文件路径
    return alice_path, bob_path, carol_path

def secret_psi(alice_path, bob_path, carol_path):
    '''
    隐私求交实现数据对齐，将求交结果保存至VDataFrame
    '''
    # 使用 v_read_csv 函数读取 CSV 文件，生成一个虚拟 DataFrame (vdf)
    vdf = v_read_csv(
        {alice: alice_path, bob: bob_path, carol: carol_path},
        spu=spu,# - spu：指定安全计算单元（SPU）
        keys='uid',# - keys='uid'：指定用于 PSI（私密集合交集）的键列
        drop_keys='uid',# - drop_keys='uid'：指定读取后要删除的键列
        psi_protocl="ECDH_PSI_3PC",# - psi_protocl="ECDH_PSI_3PC"：指定使用 ECDH_PSI_3PC 协议进行三方 PSI
    )
    # 返回生成的虚拟 DataFrame (vdf)
    return vdf

def Missing_Value_Filling(vdf):
    '''
    缺失值填充
    填充规则为：填充该列中的‘众数’
    '''
    # 定义需要填充缺失值的列名
    cols = ['workclass', 'occupation', 'native-country']
    
    # 遍历每一列
    for col in cols:
        # 找到该列的众数
        most_frequent_value = vdf[col].mode()[0]
        
        # 使用众数填充该列中的缺失值
        vdf[col].fillna(most_frequent_value, inplace=True)
    
    # 返回填充缺失值后的 DataFrame
    return vdf

def label_encode_function(vdf):
    '''
    对无序且二值的序列，采用label encoding，转换为0/1表示
    '''
    # 创建一个 LabelEncoder 对象，用于将分类数据转换为数值数据
    label_encoder = LabelEncoder()
    
    # 定义需要进行标签编码的列名
    cols = ['sex', 'income']
    
    # 遍历每一列
    for col in cols:
        # 拟合标签编码器到该列数据
        label_encoder.fit(vdf[col])
        
        # 将该列数据转换为数值数据
        vdf[col] = label_encoder.transform(vdf[col])
    
    # 返回进行标签编码后的 DataFrame
    return vdf

def Ordinal_Cate_Features(vdf):
    '''
    对于有序的类别数据，构建映射，将类别数据转换为0~n-1的整数
    '''
    vdf['education'] = vdf['education'].replace(
        {
            "Preschool":0,
            "1st-4th":1,
            "5th-6th":2,
            "7th-8th":3,
            "9th":4,
            "10th":5,
            "11th":6,
            "12th":7,
            "HS-grad":8,
            "Some-college":9,
            "Assoc-voc":10,
            "Assoc-acdm":11,
            "Bachelors":12,
            "Masters":13,
            "Prof-school":14,
            "Doctorate":15
        }
    )
    return vdf

def One_Hot_Function(vdf):
    '''
    对于无序类别数据，采用one-hot编码
    '''
    # 定义需要进行one-hot编码的列名
    onehot_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    
    # 创建一个 OneHotEncoder 对象，用于将分类数据转换为one-hot编码
    onehot_encoder = OneHotEncoder()
    
    # 拟合one-hot编码器到指定列的数据
    onehot_encoder.fit(vdf[onehot_cols])
    
    # 创建一个新的 DataFrame vdf_hat，删除需要one-hot编码的列
    vdf_hat = vdf.drop(columns=onehot_cols)
    
    # 将指定列的数据转换为one-hot编码
    enc_feats = onehot_encoder.transform(vdf[onehot_cols])
    
    # 获取one-hot编码后的特征名称
    features_name = enc_feats.columns
    
    # 从原始 DataFrame 中删除需要one-hot编码的列
    vdf = vdf.drop(columns=onehot_cols)
    
    # 将one-hot编码后的特征添加到 DataFrame 中
    vdf[features_name] = enc_feats
    
    # 返回进行one-hot编码后的 DataFrame 和删除指定列后的 DataFrame
    return vdf, vdf_hat

def standard_scaler_func(vdf):
    '''
    对数值进行标准化
    '''
    # 创建 vdf 的副本，以避免对原始数据进行修改
    vdf = vdf.copy()
    
    # 从 vdf 中删除 'income' 列，并将剩余的数据存储在 X 中
    X = vdf.drop(columns=['income'])
    
    # 将 'income' 列的数据存储在 y 中
    y = vdf['income']
    
    # 创建一个 StandardScaler 对象，用于标准化数据
    scaler = StandardScaler()
    
    # 拟合标准化器并转换 X 中的数据
    X = scaler.fit_transform(X)
    
    # 将标准化后的数据赋值回 vdf 的相应列
    vdf[X.columns] = X
    
    # 返回标准化后的 DataFrame
    return vdf

def split_train_test(vdf, train_size, random_state):
    '''
    训练集和测试集拆分
    '''
    # 使用 train_test_split 函数将数据集拆分为训练集和测试集
    train_vdf, test_vdf = train_test_split(vdf, train_size=train_size, random_state=random_state)
    
    # 从训练集中删除 'income' 列，并将剩余的数据存储在 train_X 中
    train_X = train_vdf.drop(columns=['income'])
    
    # 将训练集中的 'income' 列存储在 train_y 中
    train_y = train_vdf['income']
    
    # 从测试集中删除 'income' 列，并将剩余的数据存储在 test_X 中
    test_X = test_vdf.drop(columns=['income'])
    
    # 将测试集中的 'income' 列存储在 test_y 中
    test_y = test_vdf['income']
    
    # 返回训练集和测试集的特征和标签
    return train_X, test_X, train_y, test_y

def vdataframe_to_spu(vdf):
    '''
    将VDataFrame数据类型转换为SPUObject
    '''
    # 创建一个空列表，用于存储每个设备上的 SPU 分区
    spu_partitions = []
    
    # 遍历 vdf 的每个分区
    for device in vdf.partitions:
        # 将每个分区的数据转换为 SPU 格式，并添加到 spu_partitions 列表中
        spu_partitions.append(vdf.partitions[device].data.to(spu))
    
    # 取出第一个 SPU 分区作为基础分区
    base_partition = spu_partitions[0]
    
    # 遍历剩余的 SPU 分区
    for i in range(1, len(spu_partitions)):
        # 使用 SPU 计算，将当前基础分区与下一个 SPU 分区在轴 1 上进行拼接
        base_partition = spu(lambda x, y: jnp.concatenate([x, y], axis=1))(
            base_partition, spu_partitions[i]
        )
    
    # 返回拼接后的基础分区
    return base_partition

def convert_to_spu(train_X, test_X, train_y, test_y):
    '''
    将训练集特征数据转换为 SPU 格式
    '''
    X_train_spu = vdataframe_to_spu(train_X)
    
    # 将训练集标签数据转换为 SPU 格式
    y_train_spu = train_y.partitions[carol].data.to(spu)
    
    # 将测试集特征数据转换为 SPU 格式
    X_test_spu = vdataframe_to_spu(test_X)
    
    # 将测试集标签数据转换为 SPU 格式
    y_test_spu = test_y.partitions[carol].data.to(spu)
    
    # 返回转换后的训练集和测试集特征及标签数据
    return X_train_spu, X_test_spu, y_train_spu, y_test_spu


def data_preprocessing(full_file_path):
    '''
    数据预处理，包括加载数据集，数据集切分
    '''
    # 给各个参与方分配临时文件路径
    _, alice_path = tempfile.mkstemp()
    _, bob_path = tempfile.mkstemp()
    _, carol_path = tempfile.mkstemp()

    data = load_dataset(full_file_path)# 加载数据集
    # print(data.head()) # 查看前几行数据以了解数据集结构
    split_dataset(data,alice_path,bob_path,carol_path) # 将数据集切分为三方数据集
    return alice_path,bob_path,carol_path

def data_process(alice_path,bob_path,carol_path):
    '''
    数据处理
    '''
    vdf = secret_psi(alice_path,bob_path,carol_path) # 隐私求交实现数据对齐
    # print(vdf) # 查看求交结果
    vdf_1 = vdf.copy()# 复制求交结果
    vdf_1 = Missing_Value_Filling(vdf_1) # 缺失值填充
    vdf_1 = label_encode_function(vdf_1) # 将无序且二值的序列转换为0/1表示
    vdf_1 = Ordinal_Cate_Features(vdf_1) # 对于有序的类别数据，构建映射，将类别数据转换为0~n-1的整数
    vdf_1, vdf_hat = One_Hot_Function(vdf_1) # 对于无序类别数据，采用one-hot编码
    vdf_1 = standard_scaler_func(vdf_1) # 对数值进行标准化

    # 训练集和测试集拆分
    train_size = 0.8 # 训练集占比
    random_state = 1234 # 随机种子
    train_X,test_X,train_y,test_y = split_train_test(vdf_1,train_size,random_state)

    # 数据类型转换
    X_train_spu,X_test_spu,y_train_spu,y_test_spu = convert_to_spu(train_X,test_X,train_y,test_y)

    # 构建返回字典
    results = {
        'X_train_spu': X_train_spu,
        'X_test_spu': X_test_spu,
        'y_train_spu': y_train_spu,
        'y_test_spu': y_test_spu,
        'vdf': vdf_1,
        'vdf_hat': vdf_hat
    }
    return results

# 文件的下载和存储路径
full_file_path = './adult_data.csv'

# 配置SPU相关设备
sf.shutdown()# 关闭所有SPU设备
sf.init(['alice','bob','carol'],address='local')
aby3_config = sf.utils.testing.cluster_def(parties=['alice', 'bob', 'carol'])
spu = sf.SPU(aby3_config)
alice = sf.PYU('alice')
bob = sf.PYU('bob')
carol = sf.PYU('carol')

# 数据集加载和切分
alice_path,bob_path,carol_path = data_preprocessing(full_file_path)

'''
数据处理
这里的vdf是psi意思求交的结果
vdf_hat是数据处理之后的结果，但是不包括one-hot编码的结果
处理结果是包括X_train_spu,X_test_spu,y_train_spu,y_test_spu的字典
'''
data_dict = data_process(alice_path,bob_path,carol_path)

# 输出处理后的数据
# print(sf.reveal(data_dict['X_train_spu']))
# print(sf.reveal(data_dict['X_test_spu']))
# print(sf.reveal(data_dict['y_train_spu']))
# print(sf.reveal(data_dict['y_test_spu']))
