
# 隐语智链·深网(FlowSecureMLP)

本项目使用SecretFlow框架实现了多方安全计算，旨在通过联合数据训练，预测城市居民的年收入是否超过50k。该项目由城市内的三个政府机构合作进行，运用了多层感知器（MLP）模型来进行神经网络训练和预测。为了确保数据的准确性和可靠性，我们进行了全面的统计分析，包括多重共线性检验（VIF）和相关系数矩阵分析，并通过直方图、饼图等可视化方式呈现数据分析结果。此外，我们还开发了一个图形用户界面（GUI），使项目的操作更加简便直观。该项目的实施不仅体现了隐私保护技术在现实中的应用潜力，同时也为政府部门间的数据共享与合作提供了安全、高效的解决方案。

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
<!-- PROJECT LOGO -->
<br />
  <h3 align="center"><font size="10"><b>隐语智链·深网(FlowSecureMLP)</b></font></h3>
  <p align="center">
    一个使用SecretFlow框架和MLP模型来进行神经网络训练和预测的项目！  
    <br />
    Team:Gryffindor
    <br />
    <a href="https://github.com/YnRen22852/secretflowgryffindor"><strong>探索本项目的文档 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/YnRen22852/secretflowgryffindor">查看Demo</a>
    ·
    <a href="https://github.com/YnRen22852/secretflowgryffindor/issues">报告Bug</a>
    ·
    <a href="https://github.com/YnRen22852/secretflowgryffindor/issues">提出新特性</a>
  </p>

</p>


 
 
## 目录

- [上手指南](#上手指南)
  - [运行代码前的环境要求](#运行代码前的环境要求)
  - [配置步骤](#配置步骤)
- [功能](#功能)
- [快速开始](#快速开始)
- [演示视频](#演示视频)
- [文件目录说明](#文件目录说明)
- [使用到的框架](#使用到的框架)
- [贡献者](#贡献者)
  - [如何参与开源项目](#如何参与开源项目)
- [版权说明](#版权说明)
- [鸣谢](#鸣谢)


### **上手指南** 

#### **运行代码前的环境要求**

1. python3
2. 隐语所需要的系统环境Ubuntu/WSL2
3. 隐语所需要的conda环境  
ps：第一条和第二条在[隐语SecretFlow快速开始](https://secret-flow.antgroup.com/docs/secretflow/zh_CN/getting_started/installation.html#)中有详细步骤
以下需要在Ubuntu/WSL2中安装相应库  
4. PyQt5库
5. LibreOffice库
6. pandas库
7. numpy库
8. matplotlib库
9. seaborn库

#### **配置步骤**

配置隐语的第一和第二个环境可以从这里开始
[隐语SecretFlow快速开始](https://secret-flow.antgroup.com/docs/secretflow/zh_CN/getting_started/installation.html#)
1. PyQt5库安装
```
pip install PyQt5
```
2. LibreOffice库安装
```
sudo apt install libreoffice
```
3. pandas库安装
```
pip install pandas
```
4. numpy库安装
```
pip install numpy
```
5. matplotlib库安装
```
pip install matplotlib
```
6. seaborn库安装
```
pip install seaborn
```

### **功能**

1. 数据预处理、统计分析以及可视化  
1.1. 数据预处理  
我们主要使用了隐语SecretFlow库里的隐私求交（PSI）技术对齐数据（`secret_psi`函数），得到对齐后的数据框。然后我们对数据集进行缺失值填充（`Missing_Value_Filling`函数），标签编码（`label_encode_function`函数），有序类别特征处理（`Ordinal_Cate_Features`函数），One-Hot 编码（`One_Hot_Function`函数），标准化（`standard_scaler_func`函数），再经过从VDataFrame到SPUObject格式的数据类型转换（`vdataframe_to_spu`函数和`convert_to_spu`函数），这样可以在SPU上进行计算。  
以上步骤被汇总到数据处理总函数`data_process`函数中。  
1.2. 数据统计分析  
我们主要使用了隐语SecretFlow库里的函数做了对于数据的全表统计分析（`full_table_statistics`函数）、相关系数矩阵分析（`correlation_coefficient_matrix`函数）、VIF多重共线性检验（`VIF_calculation`函数）  
1.3. 数据可视化  
我们主要使用了seaborn库里的相关函数对数据分别做了直方图、条形图以及饼图。  
详见`data_visualize`函数  
2. 训练  
我们分别实现了在CPU上和SPU上的神经网络训练（`train_auto_grad`函数和`train_auto_grad_spu`函数），相比于在CPU上的用明文数据进行训练，我们通过使用隐语SecretFlow库的SPU，实现了在SPU上的训练。  
3. 预测  
我们分别实现了在CPU上和SPU上的神经网络预测（`predict`函数和`predict_spu`函数），相比于在CPU上的用明文数据进行预测，我们通过使用隐语SecretFlow库的SPU，实现了在SPU上的预测，这样做可以做到数据隐私保护，分布式安全计算，确保了透明性和可追溯性。  

### **快速开始** 

1. 将以上环境配置好之后，把仓库clone到本地就可以开始了
2. 打开图形话界面直接运行，之后根据图形化界面进行操作  
   ps：需要激活隐语的环境

### **演示视频**

https://github.com/user-attachments/assets/d2c20784-2f3a-430b-83b0-a6aaf30ec183


### 文件目录说明

```
filetree 
├── LICENSE
├── README.md
├── /GUI/   #图形化界面
|  ├── /neural_network_gui.py/
├── /data/  #数据处理
│  ├── /corr_coefficient_matrix.py/
│  ├── /data_process.py/
│  ├── /data_visual.py/
│  ├── /full_table_statistics.py/
│  ├── /multicollinearity_test.py/
├── /neural_network/   #神经网络模型
│  ├── /mlp.py/

```


### 使用到的框架

- [隐语SecretFlow](https://secret-flow.antgroup.com/)

### 贡献者

请阅读[贡献者](https://github.com/YnRen22852/secretflowgryffindor/graphs/contributors) 查阅为该项目做出贡献的开发者。

#### 如何参与开源项目

贡献使开源社区成为一个学习、激励和创造的绝佳场所。你所作的任何贡献都是**非常感谢**的。


1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

   
### 版权说明

该项目签署了Apache License 2.0授权许可，详情请参阅 [LICENSE](https://github.com/YnRen22852/secretflowgryffindor/blob/master/LICENSE)

### 鸣谢


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [Img Shields](https://shields.io)
- [Choose an Open Source License](https://choosealicense.com)
- [GitHub Pages](https://pages.github.com)
- [Animate.css](https://daneden.github.io/animate.css)
- [隐语](https://secret-flow.antgroup.com/)

<!-- links -->
[your-project-path]:https://github.com/YnRen22852/secretflowgryffindor
[contributors-shield]: https://img.shields.io/github/contributors/YnRen22852/secretflowgryffindor.svg?style=flat-square
[contributors-url]: https://github.com/YnRen22852/secretflowgryffindor/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/YnRen22852/secretflowgryffindor.svg?style=flat-square
[forks-url]: https://github.com/YnRen22852/secretflowgryffindor/network/members
[stars-shield]: https://img.shields.io/github/stars/YnRen22852/secretflowgryffindor.svg?style=flat-square
[stars-url]: https://github.com/YnRen22852/secretflowgryffindor/stargazers
[issues-shield]: https://img.shields.io/github/issues/YnRen22852/secretflowgryffindor.svg?style=flat-square
[issues-url]: https://github.com/YnRen22852/secretflowgryffindor/issues
[license-shield]: https://img.shields.io/github/license/YnRen22852/secretflowgryffindor.svg?style=flat-square
[license-url]: https://github.com/YnRen22852/secretflowgryffindor/blob/master/LICENSE
