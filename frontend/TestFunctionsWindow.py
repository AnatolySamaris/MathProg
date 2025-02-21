from PyQt5.QtWidgets import QDialog, QListWidget, QPushButton, QVBoxLayout
from PyQt5.QtGui import QFont

functions = {
    "Расширенная функция Розенброка": [
        r"\sum_{i=1}^{10} (100*(x_{2*i} - x_{2*i-1}^2)^2 + (1-x_{2*i-1})^2)",
        {f"x{i}": [-10, 10] for i in range(1, 20+1)}
    ],
    "Расширенная функция Розенброка, версия 2": [
        r"\sum_{i=1}^{20} (100*(x_i - x_{i-1}^2)^2 + (1-x_{i-1})^2)",
        {f"x{i}": [-10, 10] for i in range(0, 20+1)}
    ],
    "Шестигорбый верблюд": [
        r"4x_1^2 - 2.1x_1^4 + \frac{x_1^6}{3}+ x_1 x_2 - 4x_2^2 + 4x_2^4",
        {f"x{i}": [-5, 5] for i in range(1, 2+1)}
    ],
    "Функция Леви 3": [
        r"\sum_{i=1}^{5} (i* \cos((i+1)*x_1+i)) * \sum_{j=1}^{5} (j* \cos((j+1)*x_2 +j))",
        {f"x{i}": [-10, 10] for i in range(1, 2+1)}
    ],
    "Функция Леви 5": [
        r"\sum_{i=1}^{5} i*\cos((i+1)*x_1+i) * \sum_{j=1}^{5} j*\cos((j+1)*x_2 +j) + (x_1 + 1.42513)^2 + (x_2 + 0.80032)^2",
        {f"x{i}": [-10, 10] for i in range(1, 2+1)}
    ],
    "Функция Биля 1": [
        r"(1.5 - x_1 + x_1*x_2)^2 + (2.25 - x_1 + x_1*x_2^2)^2 + (2.625 - x_1 + x_1*x_2^3)^2",
        {f"x{i}": [-4.5, 4.5] for i in range(1, 2+1)}
    ],
    "Функция Бута": [
        r"(x_1 + 2*x_2 - 7)^2 (2*x_1 + x_2 - 5)^2",
        {f"x{i}": [-10, 10] for i in range(1, 2+1)}
    ],
    "Функция Матиаса": [
        r"0.26*(x_1^2 + x_2^2)-0.48*x_1*x_2",
        {f"x{i}": [-10, 10] for i in range(1, 2+1)}
    ],
    "Функция Бранина": [
        r"(x_2 - \frac{5.1}{4 * \pi^2} * x_1^2 + \frac{5}{\pi}*x_1 - 6)^2 +10*(1-\frac{1}{8*\pi})*\cos(x_1)+10",
        {"x1": [-5, 10], "x2": [0, 15]}
    ],
    "Функция Растригина": [
        r"x_1^2 + x_2^2 - \cos(12*x_1) - \cos(18*x_2)",
        {f"x{i}": [-1, 1] for i in range(1, 2+1)}
    ],
    "Функция Гриванка 2": [
        r"\frac{x_1^2 + x_2^2}{200} - \cos(x_1) * \cos(\frac{x_2}{\sqrt{2}}) + 1",
        {f"x{i}": [-100, 100] for i in range(1, 2+1)}
    ],
    "Функция Биггса Exp2": [
        r"\sum_{i=1}^{10} (\exp{-\frac{i*x_1}{10}} - 5 * \exp{-\frac{i*x_2}{10}} - \exp{-\frac{i}{10}} + 5 * \exp{i})",
        {f"x{i}": [0, 20] for i in range(1, 2+1)}
    ],
    "Функция Треккани": [
        r"x_1^4 + 4*x_1^3 + 4 *x_1^2 + x_2^2",
        {f"x{i}": [-5, 5] for i in range(1, 2+1)}
    ],
    "Трехгорбый верблюд": [
        r"2*x_1^2 - 1.05*x_1^4 + \frac{x_1^6}{6} + x_1*x_2 + x_2^2",
        {f"x{i}": [-5, 5] for i in range(1, 2+1)}
    ],
    "Функция Бранина 2": [
        r"(1 - 2*x_2 + \frac{1}{20} * \sin(4*x_2 * \pi) - x_1)^2 + (x_2 - \frac{1}{2} * \sin(2 * x_1 * \pi))^2",
        {f"x{i}": [-10, 10] for i in range(1, 2+1)}
    ]
}

class TestFunctionsWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Выбор функции")
        self.setFixedWidth(400)
        self.setFixedHeight(300)

        self.font = QFont()
        self.font.setPointSize(10)

        self.functions = functions

        layout = QVBoxLayout()

        # Создаем список функций
        self.list_widget = QListWidget()
        for key in self.functions.keys():
            self.list_widget.addItem(key)
            self.list_widget.setFont(self.font)
        # for i in range(1, 16):
            # self.list_widget.addItem(f"Функция {i}")
        self.list_widget.itemSelectionChanged.connect(self.enable_accept_button)

        layout.addWidget(self.list_widget)

        self.accept_button = QPushButton("Принять")
        self.accept_button.setFixedHeight(30)
        self.accept_button.setStyleSheet("""
            QPushButton {
                border: 1px solid black;
                background-color: white;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: lightgray;
            }
            QPushButton:pressed {
                background-color: darkgray;
            }
        """)
        self.accept_button.setEnabled(False)
        self.accept_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.setFixedHeight(30)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                border: 1px solid black;
                background-color: white;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: lightgray;
            }
            QPushButton:pressed {
                background-color: darkgray;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)

        layout.addWidget(self.accept_button)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

    def enable_accept_button(self):
        self.accept_button.setEnabled(bool(self.list_widget.selectedItems()))

    def get_selected_function(self):
        selected_item = self.list_widget.selectedItems()[0]
        func, constraints = self.functions[selected_item.text()]
        return func, constraints