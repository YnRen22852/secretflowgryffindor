# 导入必要的标准库模块
import sys          # 系统相关模块，用于获取命令行参数以及退出程序
import subprocess   # 子进程模块，用于在Python中调用外部命令

# 导入 PyQt5 库中的必要组件，用于构建图形界面
from PyQt5.QtWidgets import (
    QApplication,       # 应用程序类，管理GUI控制流和主设置
    QMainWindow,        # 主窗口类，用于创建主窗口
    QWidget,            # 通用窗口部件类，其他部件的基类
    QVBoxLayout,        # 垂直布局类，用于垂直排列子控件
    QPushButton,        # 按钮部件类，用户交互的重要组件
    QLabel,             # 标签部件类，用于显示文本或图片
    QStackedWidget,     # 堆栈窗口类，用于页面切换，多个页面互相切换显示
    QSizePolicy,        # 尺寸策略类，决定控件如何在布局中调整大小
    QTextEdit,          # 文本编辑部件类，用于显示或编辑多行文本
    QLineEdit,          # 单行文本编辑部件类，用于输入单行文本
    QHBoxLayout         # 水平布局类，用于水平排列子控件
)
from PyQt5.QtCore import Qt  # 导入Qt核心模块，用于对齐方式和其他常量设置
from PyQt5.QtGui import QFont  # 导入字体模块，用于设置控件字体

# 定义一个函数用于过滤子进程输出中的指定关键字
def filter_output(output, keywords):
    """
    过滤掉包含任意关键字的行。
    
    参数:
      output (str): 要过滤的原始输出字符串。
      keywords (list): 包含需要过滤的关键字列表。
      
    返回:
      str: 过滤后的字符串，去除了包含指定关键字的行。
    """
    # 将输出字符串按照行分割成列表
    lines = output.splitlines()
    # 使用列表推导式过滤掉包含任意一个关键字的行
    filtered_lines = [line for line in lines if not any(keyword in line for keyword in keywords)]
    # 将过滤后的行重新合并为字符串，并返回
    return '\n'.join(filtered_lines)

# 定义主窗口类，继承自 QMainWindow
class MainWindow(QMainWindow):
    def __init__(self):
        # 调用父类构造方法，初始化主窗口
        super().__init__()

        # 设置主窗口的标题为 "隐语智链·深网 (FlowSecureMLP)"
        self.setWindowTitle("隐语智链·深网 (FlowSecureMLP)")
        # 设置窗口的初始位置和尺寸 (x=100, y=100, width=1000, height=800)
        self.setGeometry(100, 100, 1000, 800)

        # 创建堆栈窗口，用于管理多个页面并实现页面切换功能
        self.stacked_widget = QStackedWidget()
        # 将堆栈窗口设置为主窗口的中心部件
        self.setCentralWidget(self.stacked_widget)

        # 创建主页面，即初始显示页面
        self.main_page = QWidget()
        # 将主页面添加到堆栈窗口中
        self.stacked_widget.addWidget(self.main_page)

        # 为主页面设置一个垂直布局管理器
        self.main_layout = QVBoxLayout(self.main_page)

        # 创建一个标签用于显示参赛题目，并将其添加到主页面
        self.topic_label = QLabel(" 隐语智链·深网 (FlowSecureMLP)", self)
        # 设置标签的文本对齐方式为居中
        self.topic_label.setAlignment(Qt.AlignCenter)
        # 设置标签的字体为黑体，大小24，粗体
        self.topic_label.setFont(QFont("黑体", 24, QFont.Bold))
        # 设置标签的尺寸策略，横向扩展，纵向最小扩展
        self.topic_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        # 将标签添加到主布局中
        self.main_layout.addWidget(self.topic_label)
        
        # 创建另一个标签用于显示队名，并将其添加到主页面
        self.team_label = QLabel("Team: Gryffindor", self)
        # 设置标签文本对齐为居中
        self.team_label.setAlignment(Qt.AlignCenter)
        # 设置标签的字体为黑体，大小20，粗体
        self.team_label.setFont(QFont("黑体", 20, QFont.Bold))
        # 设置标签的尺寸策略，同样横向扩展，纵向最小扩展
        self.team_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        # 将队名标签添加到主布局中
        self.main_layout.addWidget(self.team_label)

        # 创建两个功能按钮，用于进入不同的功能页面
        self.function1_button = QPushButton("Function 1: Data Analysis", self)
        self.function2_button = QPushButton("Function 2: Run Neural Network", self)
        # 设置按钮的字体为宋体，大小14
        self.function1_button.setFont(QFont("宋体", 14))
        self.function2_button.setFont(QFont("宋体", 14))
        # 设置按钮的尺寸策略，使按钮横向扩展，纵向最小扩展
        self.function1_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.function2_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        # 将功能按钮依次添加到主布局中
        self.main_layout.addWidget(self.function1_button)
        self.main_layout.addWidget(self.function2_button)
        
        # 连接功能按钮的点击事件，当点击时调用相应的方法显示对应页面
        self.function1_button.clicked.connect(self.show_function1)
        self.function2_button.clicked.connect(self.show_neural_network_page)
        
        # 创建 Function 1（数据分析功能）的子页面以及其内部组件
        self.create_function1_pages()

    # 定义方法，用于创建 Function 1 的子功能页面及相关组件
    def create_function1_pages(self):
        # 创建一个新的页面，用于 Function 1
        self.function1_page = QWidget()
        # 为该页面设置一个垂直布局管理器
        self.function1_layout = QVBoxLayout(self.function1_page)
        # 将 Function 1 页面添加到堆栈窗口中
        self.stacked_widget.addWidget(self.function1_page)

        # 创建四个子功能按钮，并分别设置功能描述文字
        self.sub_function1_button = QPushButton("Sub Function 1: Full Table Statistics", self)
        self.sub_function2_button = QPushButton("Sub Function 2: VIF Multicollinearity Test", self)
        self.sub_function3_button = QPushButton("Sub Function 3: Correlation Coefficient Matrix", self)
        self.sub_function4_button = QPushButton("Sub Function 4: Data Visual", self)
        # 设置每个子功能按钮的字体为宋体，大小12
        self.sub_function1_button.setFont(QFont("宋体", 12))
        self.sub_function2_button.setFont(QFont("宋体", 12))
        self.sub_function3_button.setFont(QFont("宋体", 12))
        self.sub_function4_button.setFont(QFont("宋体", 12))
        # 将所有子功能按钮依次添加到 Function 1 页面的布局中
        self.function1_layout.addWidget(self.sub_function1_button)
        self.function1_layout.addWidget(self.sub_function2_button)
        self.function1_layout.addWidget(self.sub_function3_button)
        self.function1_layout.addWidget(self.sub_function4_button)

        # 创建一个回退按钮，点击后返回主菜单
        self.back_to_main_button = QPushButton("Back to Main Menu", self)
        # 设置回退按钮的尺寸策略为最小尺寸
        self.back_to_main_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        # 将回退按钮添加到 Function 1 页面布局中
        self.function1_layout.addWidget(self.back_to_main_button)
        # 连接回退按钮的点击事件，调用 go_back_to_main 方法返回主页面
        self.back_to_main_button.clicked.connect(self.go_back_to_main)

        # 为所有子功能按钮设置扩展尺寸策略，使按钮在空间中均匀扩展
        self.sub_function1_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sub_function2_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sub_function3_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sub_function4_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 连接子功能按钮的点击事件：
        # 当点击按钮时，调用 show_output_page 方法并传递相应的脚本路径和页面标题
        self.sub_function1_button.clicked.connect(lambda: self.show_output_page("secretflowgryffindor/secretflowgryffindor/data/full_table_statistics.py", "Full Table Statistics Output"))
        self.sub_function2_button.clicked.connect(lambda: self.show_output_page("secretflowgryffindor/secretflowgryffindor/data/multicollinearity_test.py", "VIF Multicollinearity Test Output"))
        self.sub_function3_button.clicked.connect(lambda: self.show_output_page("secretflowgryffindor/secretflowgryffindor/data/corr_coefficient_matrix.py", "Correlation Coefficient Matrix Output"))
        # 对于第四个子功能，点击后显示数据可视化页面
        self.sub_function4_button.clicked.connect(self.show_data_visual_page)

    # 定义方法用于显示 Function 1 页面
    def show_function1(self):
        # 将堆栈窗口当前页面切换为 Function 1 页面
        self.stacked_widget.setCurrentWidget(self.function1_page)
    
    # 定义方法，用于显示输出页面并执行指定的脚本文件
    def show_output_page(self, script_path, page_title, main_menu=False):
        # 创建一个新的页面，用于显示脚本执行后的输出
        output_page = QWidget()
        # 为该页面设置一个垂直布局管理器
        output_layout = QVBoxLayout(output_page)
        # 将输出页面添加到堆栈窗口中
        self.stacked_widget.addWidget(output_page)
        
        # 创建一个文本编辑框用于显示输出结果，并设置为只读
        output_text = QTextEdit(output_page)
        output_text.setReadOnly(True)
        # 将文本编辑框添加到输出页面的布局中
        output_layout.addWidget(output_text)
        
        # 定义需要过滤掉的关键字列表，避免输出中出现不需要的信息
        keywords_to_filter = ["pid", "SPURuntime", "info"]
    
        try:
            # 使用 subprocess 运行指定的 Python 脚本，捕获标准输出和错误信息
            result = subprocess.run(["python", script_path], capture_output=True, text=True, check=True)
            # 对输出进行过滤处理，去除包含关键字的行
            filtered_output = filter_output(result.stdout, keywords_to_filter)
            # 将过滤后的输出添加到文本框中，并加上页面标题
            output_text.append(f"{page_title}:\n\n{filtered_output}")
        except subprocess.CalledProcessError as e:
            # 如果脚本执行出错，则过滤错误信息并显示在文本框中
            filtered_error = filter_output(e.stderr, keywords_to_filter)
            output_text.append(f"An error occurred while running {script_path}:\n\n{filtered_error}")
            
        # 创建回退按钮，根据参数决定返回到主菜单或 Function 1 菜单
        back_button = QPushButton("Back to Main Menu" if main_menu else "Back to Function 1 Menu", self)
        # 设置回退按钮的尺寸策略为最小尺寸
        back_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        # 将回退按钮添加到输出页面的布局中
        output_layout.addWidget(back_button)
        # 根据参数选择正确的回退操作
        back_button.clicked.connect(self.go_back_to_main if main_menu else self.go_back_to_function1)

        # 切换到输出页面，显示脚本执行结果
        self.stacked_widget.setCurrentWidget(output_page)

    # 定义方法用于显示数据可视化交互页面
    def show_data_visual_page(self):
        # 创建一个新的页面，用于数据可视化
        data_visual_page = QWidget()
        # 设置垂直布局管理器，用于排列页面组件
        data_visual_layout = QVBoxLayout(data_visual_page)
        # 将数据可视化页面添加到堆栈窗口中
        self.stacked_widget.addWidget(data_visual_page)

        # 创建一个标签，用于提示用户输入数据可视化选项
        input_label = QLabel("Enter a value (1-4):"+ "\n"+
 """    |   1. alice 数据可视化\t\t|
    |   2. bob 数据可视化\t\t|
    |   3. carol 数据可视化\t\t|
    |   4. full_file_path 数据可视化\t| """, self)
        # 将提示标签添加到数据可视化页面的布局中
        data_visual_layout.addWidget(input_label)
        
        # 创建一个单行文本编辑框，让用户输入选项
        input_edit = QLineEdit(self)
        data_visual_layout.addWidget(input_edit)

        # 创建一个文本编辑框，用于展示数据可视化命令执行后的结果
        result_text = QTextEdit(self)
        result_text.setReadOnly(True)
        data_visual_layout.addWidget(result_text)

        # 创建一个按钮，用于触发数据可视化操作
        run_button = QPushButton("Run Data Visual", self)
        data_visual_layout.addWidget(run_button)

        # 定义运行数据可视化操作的函数
        def run_data_visual():
            # 获取用户输入的选项值
            choice = input_edit.text()

            # 根据用户输入构造调用数据可视化脚本的命令
            dv_command = ["python", "network_demo/data_visual.py", choice]
            # 定义需要过滤的关键字，过滤掉不需要的信息
            keywords_to_filter = ["pid", "SPURuntime", "info"]
            try:
                # 执行命令并捕获输出信息
                result = subprocess.run(dv_command, capture_output=True, text=True, check=True)
                # 过滤输出内容
                filtered_output = filter_output(result.stdout, keywords_to_filter)
                # 将过滤后的结果显示在结果文本框中
                result_text.append(filtered_output)
            except subprocess.CalledProcessError as e:
                # 如果命令执行失败，则显示错误信息
                result_text.append(f"An error occurred:\n{e.stderr}")
        # 将 run_data_visual 函数与运行按钮的点击事件连接
        run_button.clicked.connect(run_data_visual)

        # 创建回退按钮，用于返回到 Function 1 菜单
        back_button = QPushButton("Back to Function 1 Menu", self)
        back_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        data_visual_layout.addWidget(back_button)
        back_button.clicked.connect(self.go_back_to_function1)

        # 切换到数据可视化页面
        self.stacked_widget.setCurrentWidget(data_visual_page)

    # 定义方法，用于显示运行神经网络功能的页面
    def show_neural_network_page(self):
        # 创建一个新的页面，用于运行神经网络
        neural_network_page = QWidget()
        # 设置垂直布局管理器
        neural_network_layout = QVBoxLayout(neural_network_page)
        # 将神经网络页面添加到堆栈窗口中
        self.stacked_widget.addWidget(neural_network_page)

        # 创建并添加标签和输入框，提示用户输入神经网络参数
        epochs_label = QLabel("Enter epochs:(默认参数需要输入2)", self)
        learning_rate_label = QLabel("Enter learning rate:(默认参数需要输入0.02)", self)
        batch_size_label = QLabel("Enter batch size:(默认参数需要输入100)", self)
        
        # 创建单行文本编辑框，用于输入 epochs 参数
        epochs_edit = QLineEdit(self)
        # 创建单行文本编辑框，用于输入 learning_rate 参数
        learning_rate_edit = QLineEdit(self)
        # 创建单行文本编辑框，用于输入 batch_size 参数
        batch_size_edit = QLineEdit(self)
        
        # 将所有标签和输入框添加到神经网络页面的布局中
        neural_network_layout.addWidget(epochs_label)
        neural_network_layout.addWidget(epochs_edit)
        neural_network_layout.addWidget(learning_rate_label)
        neural_network_layout.addWidget(learning_rate_edit)
        neural_network_layout.addWidget(batch_size_label)
        neural_network_layout.addWidget(batch_size_edit)

        # 创建一个只读的文本编辑框，用于显示神经网络运行的结果
        nn_result_text = QTextEdit(self)
        nn_result_text.setReadOnly(True)
        neural_network_layout.addWidget(nn_result_text)

        # 创建一个按钮，用于触发神经网络的运行操作
        nn_run_button = QPushButton("Run Neural Network", self)
        neural_network_layout.addWidget(nn_run_button)
        # 定义需要过滤的关键字列表
        keywords_to_filter = ["pid", "SPURuntime", "info"]
        # 定义运行神经网络操作的函数
        def run_neural_network():
            # 获取用户输入的参数
            epochs = epochs_edit.text()
            learning_rate = learning_rate_edit.text()
            batch_size = batch_size_edit.text()

            # 构造调用神经网络脚本的命令，包括参数
            nn_command = [
                "python", "network_demo/mlp.py", 
                "--epochs", epochs, 
                "--learning_rate", learning_rate, 
                "--batch_size", batch_size
            ]

            try:
                # 运行命令并捕获输出
                result = subprocess.run(nn_command, capture_output=True, text=True, check=True)
                # 对输出进行过滤，去除无关信息
                filtered_output = filter_output(result.stdout, keywords_to_filter)
                # 将过滤后的结果显示在结果文本框中
                nn_result_text.append(filtered_output)
            except subprocess.CalledProcessError as e:
                # 如果命令执行出错，则显示错误信息
                nn_result_text.append(f"An error occurred:\n{e.stderr}")
        # 将运行神经网络函数绑定到运行按钮的点击事件上
        nn_run_button.clicked.connect(run_neural_network)

        # 创建一个回退按钮，用于返回到主菜单
        back_button = QPushButton("Back to Main Menu", self)
        back_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        neural_network_layout.addWidget(back_button)
        back_button.clicked.connect(self.go_back_to_main)

        # 切换到神经网络页面，显示页面内容
        self.stacked_widget.setCurrentWidget(neural_network_page)

    # 定义方法，用于从 Function 1 页面返回
    def go_back_to_function1(self):
        # 切换堆栈窗口当前页面为 Function 1 页面
        self.stacked_widget.setCurrentWidget(self.function1_page)
    
    # 定义方法，用于返回到主菜单页面
    def go_back_to_main(self):
        # 切换堆栈窗口当前页面为主页面
        self.stacked_widget.setCurrentWidget(self.main_page)

# 程序入口，只有在直接运行此文件时才执行下面的代码
if __name__ == "__main__":
    # 创建应用程序对象，并传入命令行参数
    app = QApplication(sys.argv)
    # 创建主窗口对象
    main_window = MainWindow()
    # 显示主窗口
    main_window.show()
    # 启动应用程序事件循环，并在退出时返回退出码
    sys.exit(app.exec_())
