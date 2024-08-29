import sys
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QStackedWidget, QSizePolicy, QTextEdit, QLineEdit, QHBoxLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

def filter_output(output, keywords):
    """
    过滤掉包含任意关键字的行。
    
    :param output: 要过滤的原始输出字符串。
    :param keywords: 需要过滤的关键字列表。
    :return: 过滤后的字符串。
    """
    lines = output.splitlines()
    filtered_lines = [line for line in lines if not any(keyword in line for keyword in keywords)]
    return '\n'.join(filtered_lines)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置主窗口标题和尺寸
        self.setWindowTitle("Gryffindor - Neural Network Based on SecretFlow")
        self.setGeometry(100, 100, 1000, 800)  # 增加窗口尺寸

        # 创建堆栈窗口用于页面切换
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # 创建主页面
        self.main_page = QWidget()
        self.stacked_widget.addWidget(self.main_page)

        # 设置主页面布局
        self.main_layout = QVBoxLayout(self.main_page)

        # 显示参赛题目（占据主界面高度的30%）
        self.topic_label = QLabel(" Neural Network Based on SecretFlow", self)
        self.topic_label.setAlignment(Qt.AlignCenter)
        self.topic_label.setFont(QFont("黑体", 24, QFont.Bold))  # 保持字体为黑体，大小24
        self.topic_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.main_layout.addWidget(self.topic_label)
        
        # 显示队名（占据主界面高度的20%）
        self.team_label = QLabel("Team: Gryffindor", self)
        self.team_label.setAlignment(Qt.AlignCenter)
        self.team_label.setFont(QFont("黑体", 20, QFont.Bold))  # 保持字体为黑体，大小20
        self.team_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.main_layout.addWidget(self.team_label)

        # 创建功能按钮（占据主界面高度的50%）
        self.function1_button = QPushButton("Function 1: Data Analysis", self)
        self.function2_button = QPushButton("Function 2: Run Neural Network", self)
        self.function1_button.setFont(QFont("宋体", 14))  # 保持默认字体，仅调整大小为14
        self.function2_button.setFont(QFont("宋体", 14))  # 保持默认字体，仅调整大小为14
        self.function1_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.function2_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.main_layout.addWidget(self.function1_button)
        self.main_layout.addWidget(self.function2_button)
        
        # 连接功能按钮点击事件
        self.function1_button.clicked.connect(self.show_function1)
        self.function2_button.clicked.connect(self.show_neural_network_page)
        
        # 创建第一个功能的子功能页面
        self.create_function1_pages()

    def create_function1_pages(self):
        # 创建第一个功能页面
        self.function1_page = QWidget()
        self.function1_layout = QVBoxLayout(self.function1_page)
        self.stacked_widget.addWidget(self.function1_page)

        # 创建子功能按钮，并改名
        self.sub_function1_button = QPushButton("Sub Function 1: Full Table Statistics", self)
        self.sub_function2_button = QPushButton("Sub Function 2: VIF Multicollinearity Test", self)
        self.sub_function3_button = QPushButton("Sub Function 3: Correlation Coefficient Matrix", self)
        self.sub_function4_button = QPushButton("Sub Function 4: Data Visual", self)
        self.sub_function1_button.setFont(QFont("宋体", 12))  # 保持默认字体，仅调整大小为17
        self.sub_function2_button.setFont(QFont("宋体", 12))  # 保持默认字体，仅调整大小为17
        self.sub_function3_button.setFont(QFont("宋体", 12))  # 保持默认字体，仅调整大小为17
        self.sub_function4_button.setFont(QFont("宋体", 12))  # 保持默认字体，仅调整大小为17
        self.function1_layout.addWidget(self.sub_function1_button)
        self.function1_layout.addWidget(self.sub_function2_button)
        self.function1_layout.addWidget(self.sub_function3_button)
        self.function1_layout.addWidget(self.sub_function4_button)

        # 增加回退按钮，返回主页面，并调整按钮尺寸
        self.back_to_main_button = QPushButton("Back to Main Menu", self)
        self.back_to_main_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.function1_layout.addWidget(self.back_to_main_button)
        self.back_to_main_button.clicked.connect(self.go_back_to_main)

        # 增加按钮尺寸
        self.sub_function1_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sub_function2_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sub_function3_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sub_function4_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 连接子功能按钮点击事件
        self.sub_function1_button.clicked.connect(lambda: self.show_output_page("network_demo/full_table_statistics.py", "Full Table Statistics Output"))
        self.sub_function2_button.clicked.connect(lambda: self.show_output_page("network_demo/multicollinearity_test.py", "VIF Multicollinearity Test Output"))
        self.sub_function3_button.clicked.connect(lambda: self.show_output_page("network_demo/corr_coefficient_matrix.py", "Correlation Coefficient Matrix Output"))
        self.sub_function4_button.clicked.connect(self.show_data_visual_page)

    def show_function1(self):
        self.stacked_widget.setCurrentWidget(self.function1_page)
    
    def show_output_page(self, script_path, page_title, main_menu=False):
        # 创建输出页面
        output_page = QWidget()
        output_layout = QVBoxLayout(output_page)
        self.stacked_widget.addWidget(output_page)
        
        # 创建输出文本框
        output_text = QTextEdit(output_page)
        output_text.setReadOnly(True)
        output_layout.addWidget(output_text)
        
        keywords_to_filter = ["pid", "SPURuntime", "info"]
    
        try:
            result = subprocess.run(["python", script_path], capture_output=True, text=True, check=True)
            filtered_output = filter_output(result.stdout, keywords_to_filter)
            output_text.append(f"{page_title}:\n\n{filtered_output}")
        except subprocess.CalledProcessError as e:
            filtered_error = filter_output(e.stderr, keywords_to_filter)
            output_text.append(f"An error occurred while running {script_path}:\n\n{filtered_error}")
            
        # 增加回退按钮，返回适当的菜单，并调整按钮尺寸
        back_button = QPushButton("Back to Main Menu" if main_menu else "Back to Function 1 Menu", self)
        back_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        output_layout.addWidget(back_button)
        back_button.clicked.connect(self.go_back_to_main if main_menu else self.go_back_to_function1)

        # 显示输出页面
        self.stacked_widget.setCurrentWidget(output_page)

    def show_data_visual_page(self):
        # 创建Data Visual交互页面
        data_visual_page = QWidget()
        data_visual_layout = QVBoxLayout(data_visual_page)
        self.stacked_widget.addWidget(data_visual_page)

        # 输入框提示和输入框
        input_label = QLabel("Enter a value (1-4):"+ "\n"+
 """    |   1. alice 数据可视化\t\t|
    |   2. bob 数据可视化\t\t|
    |   3. carol 数据可视化\t\t|
    |   4. full_file_path 数据可视化\t| """, self)
        data_visual_layout.addWidget(input_label)
        
        input_edit = QLineEdit(self)
        data_visual_layout.addWidget(input_edit)

        # 结果展示框
        result_text = QTextEdit(self)
        result_text.setReadOnly(True)
        data_visual_layout.addWidget(result_text)

        # 运行按钮
        run_button = QPushButton("Run Data Visual", self)
        data_visual_layout.addWidget(run_button)

        # 运行逻辑
        def run_data_visual():
            choice = input_edit.text()

            # 根据用户输入的值调整命令
            dv_command = ["python", "network_demo/data_visual.py", choice]
            keywords_to_filter = ["pid", "SPURuntime", "info"]
            try:
                result = subprocess.run(dv_command, capture_output=True, text=True, check=True)
                filtered_output = filter_output(result.stdout, keywords_to_filter)
                result_text.append(filtered_output)
            except subprocess.CalledProcessError as e:
                result_text.append(f"An error occurred:\n{e.stderr}")
        run_button.clicked.connect(run_data_visual)

        # 回退按钮
        back_button = QPushButton("Back to Function 1 Menu", self)
        back_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        data_visual_layout.addWidget(back_button)
        back_button.clicked.connect(self.go_back_to_function1)

        # 显示页面
        self.stacked_widget.setCurrentWidget(data_visual_page)

    def show_neural_network_page(self):
        # 创建运行神经网络功能的页面
        neural_network_page = QWidget()
        neural_network_layout = QVBoxLayout(neural_network_page)
        self.stacked_widget.addWidget(neural_network_page)

        # 输入框设置
        epochs_label = QLabel("Enter epochs:(默认参数需要输入2)", self)
        learning_rate_label = QLabel("Enter learning rate:(默认参数需要输入0.02)", self)
        batch_size_label = QLabel("Enter batch size:(默认参数需要输入100)", self)
        
        epochs_edit = QLineEdit(self)
        learning_rate_edit = QLineEdit(self)
        batch_size_edit = QLineEdit(self)
        
        neural_network_layout.addWidget(epochs_label)
        neural_network_layout.addWidget(epochs_edit)
        neural_network_layout.addWidget(learning_rate_label)
        neural_network_layout.addWidget(learning_rate_edit)
        neural_network_layout.addWidget(batch_size_label)
        neural_network_layout.addWidget(batch_size_edit)

        # 结果展示框
        nn_result_text = QTextEdit(self)
        nn_result_text.setReadOnly(True)
        neural_network_layout.addWidget(nn_result_text)

        # 运行按钮
        nn_run_button = QPushButton("Run Neural Network", self)
        neural_network_layout.addWidget(nn_run_button)
        keywords_to_filter = ["pid", "SPURuntime", "info"]
        # 运行逻辑
        def run_neural_network():
            epochs = epochs_edit.text()
            learning_rate = learning_rate_edit.text()
            batch_size = batch_size_edit.text()

            nn_command = [
                "python", "network_demo/mlp.py", 
                "--epochs", epochs, 
                "--learning_rate", learning_rate, 
                "--batch_size", batch_size
            ]

            try:
                result = subprocess.run(nn_command, capture_output=True, text=True, check=True)
                filtered_output = filter_output(result.stdout, keywords_to_filter)
                nn_result_text.append(filtered_output)
            except subprocess.CalledProcessError as e:
                nn_result_text.append(f"An error occurred:\n{e.stderr}")
        nn_run_button.clicked.connect(run_neural_network)

        # 回退按钮
        back_button = QPushButton("Back to Main Menu", self)
        back_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        neural_network_layout.addWidget(back_button)
        back_button.clicked.connect(self.go_back_to_main)

        # 显示页面
        self.stacked_widget.setCurrentWidget(neural_network_page)

    def go_back_to_function1(self):
        self.stacked_widget.setCurrentWidget(self.function1_page)
    
    def go_back_to_main(self):
        self.stacked_widget.setCurrentWidget(self.main_page)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
