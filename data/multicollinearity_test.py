from secretflow.stats.ss_vif_v import VIF
from data_process import data_dict,spu

def VIF_calculation(vdf_hat):
    '''
    计算VIF来进行多重共线性检验
    '''
    # 创建一个 VIF 计算器对象，传入 SPU 作为参数
    vif_calculator = VIF(spu)
    
    # 使用 VIF 计算器计算 vdf_hat 数据集的 VIF 值
    vif_results = vif_calculator.vif(vdf_hat)
    
    # 打印分隔符和多重共线性检验标题
    print("="*40 + "多重共线性检验" + "="*40)
    
    # 打印 vdf_hat 数据集的列名
    print(vdf_hat.columns)
    
    # 打印计算得到的 VIF 结果
    print(vif_results)

if __name__ == '__main__':
    vdf_hat = data_dict['vdf_hat']
    VIF_calculation(vdf_hat)
