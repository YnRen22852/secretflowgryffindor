
# Neural Network Based on SecretFlow

本项目使用SecretFlow框架实现了多方安全计算，旨在通过联合数据训练，预测城市居民的年收入是否超过50k。该项目由城市内的三个政府机构合作进行，运用了多层感知器（MLP）模型来进行神经网络训练和预测。为了确保数据的准确性和可靠性，我们进行了全面的统计分析，包括多重共线性检验（VIF）和相关系数矩阵分析，并通过直方图、饼图等可视化方式呈现数据分析结果。此外，我们还开发了一个图形用户界面（GUI），使项目的操作更加简便直观。该项目的实施不仅体现了隐私保护技术在现实中的应用潜力，同时也为政府部门间的数据共享与合作提供了安全、高效的解决方案。

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/YnRen22852/secretflowgryffindor">
    <img src="https://img2024.cnblogs.com/blog/3248951/202408/3248951-20240829124619808-424781544.webp" alt="Logo" width="400" height="400">
  </a>

  <h3 align="center">"Neural Network Based on SecretFlow</h3>
  <p align="center">
    一个使用SecretFlow、MLP模型来进行神经网络训练和预测的项目！
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
- [快速开始](#快速开始)
- [文件目录说明](#文件目录说明)
- [使用到的框架](#使用到的框架)
- [贡献者](#贡献者)
  - [如何参与开源项目](#如何参与开源项目)
- [版权说明](#版权说明)
- [鸣谢](#鸣谢)


## **上手指南** 
### **运行代码前的环境要求**

1. python3
2. 隐语所需要的系统环境Ubuntu/WSL2
3. 隐语所需要的conda环境
ps：第一条和第二条在[隐语SecretFlow快速开始](https://secret-flow.antgroup.com/docs/secretflow/zh_CN/getting_started/installation.html#)中有详细步骤
5. PyQt5库
6. LibreOffice库
7. pandas库
8. numpy库
9. matplotlib库

### **配置步骤**

配置下列隐语的第一和第二个环境可以从这里开始
[隐语SecretFlow快速开始](https://secret-flow.antgroup.com/docs/secretflow/zh_CN/getting_started/installation.html#)
1. PyQt5库安装
```pip install PyQt5```
2. LibreOffice库安装
```sudo apt install libreoffice```
3. pandas库安装
```pip install pandas```
5. numpy库安装
```pip install numpy```
7. matplotlib库安装
```pip install matplotlib```

### **快速开始** 

1. 将以上环境配置好之后，把仓库clone到本地就可以开始了
2. 打开图形话界面直接运行
   
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
