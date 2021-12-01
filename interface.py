from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

from plr import parametric_programming, linear_programming, linear_fractional_programming


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(500, 500)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        MainWindow.setWindowTitle("MainWindow")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 50, 300, 51))
        self.label.setMinimumSize(QtCore.QSize(300, 0))
        self.label.setStyleSheet("font: 22pt \"Andale Mono\";\n"
                                 "color:rgb(43,59,78);")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label.setText("Linear programming")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(30, 160, 128, 131))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.input_file_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.input_file_btn.setMinimumSize(QtCore.QSize(0, 26))
        self.input_file_btn.setStyleSheet("QPushButton {\n"
                                          "    background-color: rgb(253, 128, 8);\n"
                                          "    color:rgb(243,247,254);\n"
                                          "    border: 0px solid;\n"
                                          "    border-radius: 12px;\n"
                                          "    font: 13pt \"Andale Mono\";\n"
                                          "}\n"
                                          "\n"
                                          "QPushButton:hover {\n"
                                          "    background-color: rgb(254, 204, 102);\n"
                                          "    color: rgb(255, 255, 255);\n"
                                          "}")
        self.input_file_btn.setObjectName("input_file_btn")
        self.verticalLayout.addWidget(self.input_file_btn)
        self.output_file_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.output_file_btn.setMinimumSize(QtCore.QSize(0, 26))
        self.output_file_btn.setStyleSheet("QPushButton {\n"
                                           "    background-color: rgb(253, 128, 8);\n"
                                           "    color:rgb(243,247,254);\n"
                                           "    border: 0px solid;\n"
                                           "    border-radius: 12px;\n"
                                           "    font: 13pt \"Andale Mono\";\n"
                                           "}\n"
                                           "\n"
                                           "QPushButton:hover {\n"
                                           "    background-color: rgb(254, 204, 102);\n"
                                           "    color: rgb(255, 255, 255);\n"
                                           "}")
        self.output_file_btn.setObjectName("output_file_btn")
        self.output_file_btn.setText("Set output file")
        self.input_file_btn.setText("Set input file")
        self.verticalLayout.addWidget(self.output_file_btn)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(200, 160, 261, 131))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.input_file_name = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.input_file_name.setMinimumSize(QtCore.QSize(0, 26))
        self.input_file_name.setStyleSheet("QLineEdit {\n"
                                           "    border: 1px solid rgb(179, 179, 179);\n"
                                           "    border-radius:10px\n"
                                           "}")
        self.input_file_name.setObjectName("lineEdit")
        self.verticalLayout_2.addWidget(self.input_file_name)
        self.output_file_name = QtWidgets.QLineEdit(self.verticalLayoutWidget_2)
        self.output_file_name.setMinimumSize(QtCore.QSize(0, 26))
        self.output_file_name.setStyleSheet("QLineEdit {\n"
                                            "    border: 1px solid rgb(179, 179, 179);\n"
                                            "    border-radius:10px\n"
                                            "}")
        self.output_file_name.setObjectName("lineEdit_2")
        self.verticalLayout_2.addWidget(self.output_file_name)
        self.solve_btn = QtWidgets.QPushButton(self.centralwidget)
        self.solve_btn.setGeometry(QtCore.QRect(412, 320, 113, 26))
        self.solve_btn.setMinimumSize(QtCore.QSize(0, 26))
        self.solve_btn.setMaximumSize(QtCore.QSize(50, 26))
        self.solve_btn.setStyleSheet("QPushButton {\n"
                                     "    background-color: rgb(253, 128, 8);\n"
                                     "    color:rgb(243,247,254);\n"
                                     "    border: 0px solid;\n"
                                     "    border-radius: 12px;\n"
                                     "    font: 13pt \"Andale Mono\";\n"
                                     "}\n"
                                     "\n"
                                     "QPushButton:hover {\n"
                                     "    background-color: rgb(254, 204, 102);\n"
                                     "    color: rgb(255, 255, 255);\n"
                                     "}")
        self.solve_btn.setObjectName("solve_btn")
        self.solve_btn.setText("Solve")
        self.logging_checkbox = QtWidgets.QCheckBox(self.centralwidget)
        self.logging_checkbox.setGeometry(QtCore.QRect(70, 350, 87, 20))
        self.logging_checkbox.setStyleSheet("font: 15pt \"Andale Mono\";\n"
                                            "color:rgb(43,59,78);")
        self.logging_checkbox.setObjectName("logging_checkbox")
        self.logging_checkbox.setText("Logging")
        self.task_cb = QtWidgets.QComboBox(self.centralwidget)
        self.task_cb.setGeometry(QtCore.QRect(200, 320, 141, 26))
        self.task_cb.setMinimumSize(QtCore.QSize(200, 26))
        self.task_cb.setMaximumSize(QtCore.QSize(400, 26))
        self.task_cb.setStyleSheet("QComboBox {\n"
                                   "    border: 1px solid rgb(179, 179, 179);\n"
                                   "    border-radius: 5px;\n"
                                   "    font: 13pt \"Andale Mono\";\n"
                                   "    color:rgb(43,59,78);\n"
                                   "}\n"
                                   "\n"
                                   "QComboBox::drop-down \n"
                                   "{\n"
                                   "    border: 0px;\n"
                                   "}")
        self.task_cb.setObjectName("task_cb")
        self.task_cb.addItems(['linear programming', 'parametric programming', 'linear fractional programming'])
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(69, 320, 91, 20))
        self.label_2.setStyleSheet("font: 15pt \"Andale Mono\";\n"
                                   "color:rgb(43,59,78);")
        self.label_2.setObjectName("label_2")
        self.label_2.setText("Task type:")
        MainWindow.setCentralWidget(self.centralwidget)

        self.input_file_btn.clicked.connect(self.set_input_file)
        self.output_file_btn.clicked.connect(self.set_output_file)
        self.solve_btn.clicked.connect(self.solve)

    def set_input_file(self):
        file_name = QFileDialog.getOpenFileName()
        self.input_file_name.setText(file_name[0])

    def set_output_file(self):
        file_name = QFileDialog.getSaveFileName()
        self.output_file_name.setText(file_name[0])

    def solve(self):
        match self.task_cb.currentText():
            case 'parametric programming':
                parametric_programming(input_file_name=self.input_file_name.text(),
                                       output_file_name=self.output_file_name.text(),
                                       include_logging=self.logging_checkbox.isChecked())
            case 'linear programming':
                linear_programming(input_file_name=self.input_file_name.text(),
                                   output_file_name=self.output_file_name.text())
            case 'linear fractional programming':
                print("There")
                linear_fractional_programming(input_file_name=self.input_file_name.text(),
                                              output_file_name=self.output_file_name.text(),
                                              include_logging=self.logging_checkbox.isChecked())
        self.input_file_name.setText("")
        self.output_file_name.setText("")
