from data_process import data_dict,spu
from secretflow.stats.ss_pearsonr_v import PearsonR
import numpy as np

def correlation_coefficient_matrix(vdf_hat):
    '''
    计算相关系数矩阵(利用皮尔逊相关系数)
    在计算相关系数矩阵的时候要排除无序类别的数据
    '''
    # 创建一个 PearsonR 对象，用于计算相关系数矩阵
    pearson_r = PearsonR(spu)
    
    # 使用 PearsonR 对象计算 vdf_hat 的相关系数矩阵
    corr_matrix = pearson_r.pearsonr(vdf_hat)
    
    # 打印标题，表示接下来输出的是相关系数矩阵
    print("==================== 相关系数矩阵 ====================\n")
    
    # 设置 numpy 的打印选项，格式化浮点数输出为小数点后三位
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    
    # 打印相关系数矩阵
    print(corr_matrix)

if __name__ == '__main__':
    vdf_hat = data_dict['vdf_hat']
    correlation_coefficient_matrix(vdf_hat) 
