from secretflow.stats.table_statistics import table_statistics
from data_process import data_dict
import os
import tempfile
import subprocess

def full_table_statistics(vdf):
    '''
    全表统计并保存为Excel文件
    '''
    # 生成全表统计数据
    data_stats = table_statistics(vdf)
    
    # 创建临时文件路径
    temp_dir = tempfile.gettempdir()  # 获取系统临时目录
    output_file = os.path.join(temp_dir, 'full_table_stats.xlsx')  # 设置文件路径

    # 将数据保存为Excel文件
    data_stats.to_excel(output_file, index=False)

    print(f"统计结果已保存至 {output_file}")

    # 打开生成的Excel文件
    try:  
        # 使用 LibreOffice Calc 打开 Excel 文件  
        subprocess.run(['libreoffice', '--calc', output_file], check=True)  
    except subprocess.CalledProcessError as e:  
        print(f"An error occurred while opening the Excel file: {e}")  
    # 对于非Windows系统，可以使用以下代码：
    # os.system(f'open "{output_file}"')  # macOS
    os.system(f'xdg-open "{output_file}"')  # Linux
    return

if __name__ == '__main__':
    full_table_statistics(data_dict['vdf'])
