import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import tempfile
import subprocess
from data_process import alice_path, bob_path, carol_path, full_file_path
from mlp import install_package

# 安装seaborn包，用于数据可视化
# install_package('seaborn')

def histogram(column_data, pdf_pages):
    '''
    绘制直方图
    '''
    plt.figure(figsize=(10, 6))
    
    # 使用 seaborn 库绘制直方图，包含核密度估计（kde），并设置直方图的柱数为 30
    sns.histplot(column_data, kde=True, bins=30)
    
    # 设置图表标题，标题内容为 'Histogram of ' 加上列名
    plt.title(f'Histogram of {column_data.name}')
    
    # 设置 x 轴标签，标签内容为列名
    plt.xlabel(column_data.name)
    
    # 设置 y 轴标签，标签内容为 'Frequency'
    plt.ylabel('Frequency')
    
    # 调整 x 轴刻度标签的旋转角度为 90 度
    plt.xticks(rotation=90)
    
    # 保存当前图表到 PDF 文件
    pdf_pages.savefig()
    
    # 关闭当前图表，释放内存
    plt.close()

def bar_chart(column_data, pdf_pages):
    '''
    绘制条形图
    '''
    plt.figure(figsize=(10, 6))
    
    # 使用 seaborn 库绘制条形图，y 轴为 column_data，颜色调色板为 'viridis'
    ax = sns.countplot(y=column_data, palette='viridis')
    
    # 设置图表标题，标题内容为 'Bar Chart of ' 加上列名
    plt.title(f'Bar Chart of {column_data.name}')
    
    # 设置 x 轴标签，标签内容为 'Count'
    plt.xlabel('Count')
    
    # 设置 y 轴标签，标签内容为列名
    plt.ylabel(column_data.name)
    
    # 设置条形标签
    for p in ax.patches:
        # 获取条形的宽度（即计数值）
        width = p.get_width()
        
        # 获取条形的高度
        height = p.get_height()
        
        # 获取条形的 x 坐标
        x = p.get_x() + width
        
        # 获取条形的 y 坐标，条形的中心位置
        y = p.get_y() + height / 2
        
        # 确定标签位置以避免重叠，将标签向右偏移一些
        label_x = x + 0.1
        label_y = y
        
        # 添加标签，显示条形的宽度（即计数值）
        ax.annotate(f'{width}', 
                    (label_x, label_y),
                    ha='left', va='center', fontsize=10, color='black')
    
    # 保存当前图表到 PDF 文件
    pdf_pages.savefig()
    
    # 关闭当前图表，释放内存
    plt.close()

def pie_chart(column_data, colors, pdf_pages):
    '''
    绘制饼图
    '''
    plt.figure(figsize=(8, 8))
    
    # 计算每个类别的频数
    sizes = column_data.value_counts()
    
    # 绘制饼图
    sizes.plot.pie(
        autopct='%1.1f%%',  # 设置百分比格式
        startangle=90,  # 设置饼图的起始角度
        colors=colors,  # 设置饼图的颜色
        wedgeprops=dict(edgecolor='grey')  # 设置切片的边缘颜色
    )
    
    # 设置图表标题，标题内容为 'Pie Chart of ' 加上列名，字体大小为 20
    plt.title(f'Pie Chart of {column_data.name}', fontsize=20)
    
    # 隐藏 y 轴标签
    plt.ylabel('')
    
    # 保存当前图表到 PDF 文件
    pdf_pages.savefig()
    
    # 关闭当前图表，释放内存
    plt.close()

def data_visualize(user_file_path):
    '''
    数据可视化
    '''
    # 定义需要绘制直方图的列名
    cols_histogram = ['age', 'fnlwgt', 'education', 'hours-per-week', 'capital-gain', 'capital-loss', 'native-country']
    
    # 定义需要绘制条形图的列名
    cols_bar_chart = ['race', 'workclass', 'occupation', 'education-num', 'marital-status', 'relationship']
    
    # 定义需要绘制饼图的列名
    cols_pie_chart = ['sex', 'income']
    
    # 读取用户提供的 CSV 文件，生成 DataFrame
    df = pd.read_csv(user_file_path)
    
    # 创建临时 PDF 文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
        # 获取临时文件的路径
        temp_file_path = temp_pdf.name
        print()
        
        # 使用 PdfPages 创建一个 PDF 文件对象
        with PdfPages(temp_file_path) as pdf_pages:
            # 遍历 DataFrame 中的每一列
            for col in df.columns:
                # 如果列名在直方图列名列表中，则绘制直方图
                if col in cols_histogram:
                    histogram(df[col], pdf_pages)
                
                # 如果列名在条形图列名列表中，则绘制条形图
                if col in cols_bar_chart:
                    bar_chart(df[col], pdf_pages)
                
                # 如果列名在饼图列名列表中，则绘制饼图
                if col in cols_pie_chart:
                    # 自定义颜色调色板
                    colors = sns.color_palette('Set2', n_colors=len(df[col].value_counts()))
                    pie_chart(df[col], colors, pdf_pages)
    
    # 尝试打开生成的 PDF 文件
    try:
        # 使用系统默认的 PDF 查看器打开生成的 PDF 文件
        subprocess.run(['evince', temp_file_path], check=True)
    except subprocess.CalledProcessError as e:
        # 如果打开文件时发生错误，打印错误信息
        print(f"An error occurred while opening the PDF file: {e}")

if __name__ == '__main__':
    choice = sys.argv[1]
    if choice == '1':  
        data_visualize(alice_path)
    elif choice == '2':  
        data_visualize(bob_path)
    elif choice == '3':  
        data_visualize(carol_path)
    elif choice == '4':  
        data_visualize(full_file_path)
    else:  
        print("输入选项有误，请重新输入！")
