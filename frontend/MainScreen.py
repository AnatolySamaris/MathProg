from PyQt5.QtWidgets import (
    QMainWindow, QWidget, 
    QVBoxLayout, QHBoxLayout, QLabel, 
    QDesktopWidget, QFrame, QLineEdit, 
    QComboBox, QTableWidget, QTableWidgetItem, 
    QSizePolicy, QHeaderView, QPushButton
)
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from functools import partial
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp
from PyQt5.QtWidgets import QStyledItemDelegate

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from backend.Function import Function
from backend.Optimizator import Optimizator

from time import time

class DoubleValidatorDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def createEditor(self, parent, option, index):
        """
        Создаем редактор (QLineEdit) с валидатором для ввода чисел.
        """
        editor = super().createEditor(parent, option, index)
        validator = QDoubleValidator()  # Валидатор для чисел с плавающей запятой
        validator.setNotation(QDoubleValidator.ScientificNotation)  # Разрешаем научную нотацию
        editor.setValidator(validator)
        return editor

class MainScreen(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "Учебно-исследовательский проект"
        self.width = 1200
        self.height = 800
        self.background_color = 'white'

        self.setWindowTitle(self.title)
        self.setGeometry(150, 150, self.width, self.height)
        self.setStyleSheet(f"background-color: {self.background_color};")
        self.center()

        # шрифт
        font = QFont()
        font.setPointSize(10)

        # валидаторы для полей ввода
        bounds_validator = QIntValidator()
        function_regexp = QRegExp("[^а-яА-Я]+")
        function_validator = QRegExpValidator(function_regexp)

        # главный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # горизонтальный layout для разделения на левую и правую части
        main_layout = QHBoxLayout(central_widget)

        # =========================================================================
        # ЛЕВАЯ СТОРОНА
        # =========================================================================

        # левая часть
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # верхняя часть левой стороны (35%)
        top_left_widget = QWidget()
        top_left_layout = QVBoxLayout(top_left_widget)  # Layout для элементов в top_left_widget

        # cлово "Функция"
        function_label = QLabel("Функция:")
        function_label.setFont(font)
        top_left_layout.addWidget(function_label)

        # поле ввода для функции
        function_input_layout = QHBoxLayout()
        fx_label = QLabel("f(x) =")
        fx_label.setFont(font)
        self.function_input = QLineEdit()
        self.function_input.setFont(font)
        self.function_input.setValidator(function_validator)
        self.function_input.textChanged.connect(self.update_beautiful_function)
        function_input_layout.addWidget(fx_label)
        function_input_layout.addWidget(self.function_input)
        top_left_layout.addLayout(function_input_layout)

        # слова "Ограничение для xi"
        constraint_label = QLabel("<p>Ограничение для x<sub>i</sub></p>")
        constraint_label.setFont(font)
        top_left_layout.addWidget(constraint_label)

        # два поля ввода для ограничений
        constraint_input_layout = QHBoxLayout()
        self.lower_bound_input = QLineEdit()
        self.lower_bound_input.setFont(font)
        self.lower_bound_input.setValidator(bounds_validator)
        xi_label = QLabel("<p> &lt;= x<sub>i</sub> &lt;= </p>")
        xi_label.setFont(font)
        self.upper_bound_input = QLineEdit()
        self.upper_bound_input.setFont(font)
        self.upper_bound_input.setValidator(bounds_validator)
        constraint_input_layout.addWidget(self.lower_bound_input)
        constraint_input_layout.addWidget(xi_label)
        constraint_input_layout.addWidget(self.upper_bound_input)
        top_left_layout.addLayout(constraint_input_layout)

        # сообщение об ошибке красным цветом
        self.error_message_label = QLabel()
        self.error_message_label.setFont(font)
        palette = QPalette()
        palette.setColor(QPalette.WindowText, QColor("red"))
        self.error_message_label.setPalette(palette)
        top_left_layout.addWidget(self.error_message_label)
        # self.error_message_label.hide() # Скрываем строку до появления ошибки

        left_layout.addWidget(top_left_widget, stretch=35)

        # разделитель между верхней и нижней частью левой стороны
        left_separator1 = QFrame()
        left_separator1.setFrameShape(QFrame.HLine)  # Горизонтальная линия
        left_layout.addWidget(left_separator1)

        # нижняя часть левой стороны (остальное)
        bottom_left_widget = QWidget()
        bottom_left_layout = QVBoxLayout(bottom_left_widget)

        # заголовок для глобальных методов
        glob_method_label = QLabel("Метод глобальной оптимизации")
        glob_method_label.setFont(font)
        bottom_left_layout.addWidget(glob_method_label)

        # выпадающий список для глобальных методов
        self.glob_methods = QComboBox()
        self.glob_methods.addItem("Метод Монте-Карло")
        self.glob_methods.addItem("Метод имитации отжига")
        self.glob_methods.setFixedHeight(40)
        self.glob_methods.setFont(font)
        bottom_left_layout.addWidget(self.glob_methods)

        # таблица для глобальных методов
        self.glob_table = QTableWidget()
        self.glob_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.glob_table.setShowGrid(True)
        self.glob_table.setFont(font)
        bottom_left_layout.addWidget(self.glob_table, stretch=2) 
        self.update_table(self.glob_methods, self.glob_table)

        self.glob_methods.currentIndexChanged.connect(
            partial(self.update_table, self.glob_methods, self.glob_table)
        )

        # заголовок для локальных методов
        loc_method_label = QLabel("Метод локальной оптимизации")
        loc_method_label.setFont(font)
        bottom_left_layout.addWidget(loc_method_label)

        # выпадающий список для локальных методов
        self.loc_methods = QComboBox()
        self.loc_methods.addItem("Метод Нелдера-Мида")
        self.loc_methods.addItem("Метод Пауэлла")
        self.loc_methods.addItem("Градиентный спуск")
        self.loc_methods.addItem("Метод Ньютона")
        self.loc_methods.addItem("BFGS")
        self.loc_methods.setFixedHeight(40)
        self.loc_methods.setFont(font)
        bottom_left_layout.addWidget(self.loc_methods)

        # таблица для локальных методов
        self.loc_table = QTableWidget()
        self.loc_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.loc_table.setShowGrid(True)
        self.loc_table.setFont(font)
        # self.loc_table.setStyleSheet("""
        #     QHeaderView::section {
        #         border: 1px solid gray;
        #     }
        # """)
        bottom_left_layout.addWidget(self.loc_table, stretch=2) 
        self.update_table(self.loc_methods, self.loc_table)

        self.loc_methods.currentIndexChanged.connect(
            partial(self.update_table, self.loc_methods, self.loc_table)
        )

        self.button_calculate = QPushButton()
        self.button_calculate.setText("Расчитать")
        self.button_calculate.setMinimumHeight(32)
        self.button_calculate.setStyleSheet("""
            QPushButton {
                border: 1px solid black;
                background-color: white;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: lightgray;
            }
            QPushButton:pressed {
                background-color: darkgray;
            }
        """)
        self.button_calculate.clicked.connect(self.calculation)
        bottom_left_layout.addWidget(self.button_calculate)

        # добавляем нижнюю часть в левую часть
        left_layout.addWidget(bottom_left_widget, stretch=65)

        # добавляем левую часть в главный layout
        main_layout.addWidget(left_widget, stretch=50)

        # разделитель между левой и правой частью
        main_separator = QFrame()
        main_separator.setFrameShape(QFrame.VLine)  # Вертикальная линия
        main_layout.addWidget(main_separator)

        # =========================================================================
        # ПРАВАЯ СТОРОНА
        # =========================================================================

        # правая сторона
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # верхняя часть правой стороны
        top_right_widget = QWidget()
        top_right_layout = QVBoxLayout(top_right_widget)

        # Виджет красивого отображения формулы
        formula_widget = QWidget()
        formula_layout = QVBoxLayout(formula_widget)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        formula_layout.addWidget(self.canvas)
        formula_layout.addStretch()

        # Добавляем красивое отображение формулы в правую верхнюю часть
        top_right_layout.addWidget(formula_widget)

        # Виджет результатов
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        # Результаты
        self.result_time_label = QLabel()
        self.result_time_label.setStyleSheet("font-size: 18px;")
        self.result_time_label.hide()
        results_layout.addWidget(self.result_time_label)

        self.result_value_label = QLabel()
        self.result_value_label.setStyleSheet("font-size: 18px;")
        self.result_value_label.hide()
        results_layout.addWidget(self.result_value_label)
        
        self.result_point_label = QLabel()
        self.result_point_label.setStyleSheet("font-size: 18px;")
        self.result_point_label.hide()
        results_layout.addWidget(self.result_point_label)

        self.result_point = QLabel()
        self.result_point.setText("")
        self.result_point.setStyleSheet("font-size: 18px;")
        self.result_point.hide()
        results_layout.addWidget(self.result_point)
        
        # Добавляем вертикальный отступ чтобы прижать все элементы к верху
        results_layout.addStretch()

        # Добавляем результаты в правую верхнюю часть
        top_right_layout.addWidget(results_widget)

        # Добавляем правую верхнюю часть в правую сторону
        right_layout.addWidget(top_right_widget, stretch=40)

        # разделитель между верхней и нижней частью правой стороны
        right_separator1 = QFrame()
        right_separator1.setFrameShape(QFrame.HLine)  # Горизонтальная линия
        right_layout.addWidget(right_separator1)

        # нижняя часть правой стороны
        bottom_right_widget = QWidget()
        bottom_right_layout = QVBoxLayout(bottom_right_widget)

        # Добавляем правую нижнюю часть в правую сторону
        right_layout.addWidget(bottom_right_widget, stretch=60)

        # добавляем правую сторону в главный layout
        main_layout.addWidget(right_widget, stretch=50)

    def center(self) -> None:
        """
        Устанавливает окно по центру экрана.
        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # общие настройки таблиц
    def set_table_parameters(self, table, headers):
        # убираем индексы строк
        table.verticalHeader().setVisible(False)

        # растягиваем таблицу на всю высоту
        table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # настраиваем растяжение заголовков на всю ширину
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        # стили для заголовков
        table.setStyleSheet("""
            QHeaderView::section {
                font-weight: bold;
                background-color: white;
            }
        """)

        delegate = DoubleValidatorDelegate(table)
        table.setItemDelegate(delegate)

        # делаем текст по центру ячеек
        table.setRowCount(1)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        for row in range(table.rowCount()):
            for col in range(table.columnCount()):
                item = QTableWidgetItem("")
                # item.setData(Qt.EditRole, 0.0) # set default value 0.0
                item.setTextAlignment(Qt.AlignCenter)
        table.setItem(row, col, item)

    def update_table(self, methods, table):
        """
        Обновляет таблицу в зависимости от выбранного метода.
        """
        selected_method = methods.currentText()

        if selected_method == "Метод Монте-Карло":
            headers = ["N"]
            self.set_table_parameters(table, headers)

        elif selected_method == "Метод имитации отжига":
            headers = ["Tₘₐₓ", "L", "r", "ε"]
            self.set_table_parameters(table, headers)

        elif selected_method == "Метод Нелдера-Мида" or selected_method == "Метод Пауэлла":
            headers = ["N", "ε"]
            self.set_table_parameters(table, headers)

        elif selected_method == "Метод Ньютона" or selected_method == "BFGS":
            headers = ["N", "ε", "h"]
            self.set_table_parameters(table, headers)

        elif selected_method == "Градиентный спуск":
            headers = ["N", "ε", "h", "λ"]
            self.set_table_parameters(table, headers)

    def update_beautiful_function(self):
        latex_text = self.function_input.text()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        try:
            func = Function(latex_text)
            latex_expr = func.get_latex_func()
            ax.text(0.5, 0.5, f'${latex_expr}$', fontsize=15, ha='center', va='center')
        except Exception as e:
            ax.text(0.5, 0.5, 'Ошибка ввода', fontsize=15, ha='center', va='center', color='red')
        self.canvas.draw()
    
    def reset_output(self): # TODO : ДОБАВИТЬ ОЧИСТКУ/УДАЛЕНИЕ ГРАФИКА
        self.result_time_label.setText("")
        self.result_time_label.hide()

        self.result_value_label.setText("")
        self.result_value_label.hide()

        self.result_point_label.setText("")
        self.result_point_label.hide()

        self.result_point.setText("")
        self.result_point.hide()

        self.error_message_label.setText("")

    def validate_inputs(self) -> bool:
        if len(self.function_input.text().strip()) == 0:
            self.error_message_label.setText("Функция не введена!")
            return False
        if len(self.lower_bound_input.text().strip()) == 0:
            self.error_message_label.setText("Нижняя граница xi не введена!")
            return False
        if len(self.upper_bound_input.text().strip()) == 0:
            self.error_message_label.setText("Верхняя граница xi не введена!")
            return False
        for col in range(self.glob_table.columnCount()):
            if (self.glob_table.item(0, col).text().strip()) == 0:
                self.error_message_label.setText("Глобальный метод не инициализирован!")
                return False
        for col in range(self.loc_table.columnCount()):
            if (self.loc_table.item(0, col).text().strip()) == 0:
                self.error_message_label.setText("Локальный метод не инициализирован!")
                return False
        return True

    def calculation(self):
        self.reset_output()

        if not self.validate_inputs():
            return

        # Парсим введенную функцию
        try:
            func = Function(self.function_input.text())
            n_vars = func.count_vars()
        except Exception:
            self.error_message_label.setText("Ошибка при считывании функции!")
            return
        
        # Получаем ограничения по xi
        lower_x = float(self.lower_bound_input.text().strip())
        upper_x = float(self.upper_bound_input.text().strip())

        # Получаем инициализирующие переменные из таблицы глобальной оптимизации
        match self.glob_methods.currentText():
            case "Метод Монте-Карло":
                N = int(self.glob_table.item(0, 0).text())  # N
            case "Метод имитации отжига":
                Tmax = float(self.glob_table.item(0, 0).text().replace(",", "."))  # Tmax
                L = int(self.glob_table.item(0, 1).text())  # L
                r = float(self.glob_table.item(0, 2).text().replace(",", "."))  # r
                eps = float(self.glob_table.item(0, 3).text().replace(",", "."))    # eps
            case _:
                return

        # Получаем инициализирующие переменные из таблицы локальной оптимизации
        match self.loc_methods.currentText():
            case "Метод Нелдера-Мида":
                N_loc = int(self.loc_table.item(0, 0).text())  # N
                eps_loc = float(self.loc_table.item(0, 1).text().replace(",", ".")) # eps
            case "Метод Пауэлла":
                N_loc = int(self.loc_table.item(0, 0).text())  # N
                eps_loc = float(self.loc_table.item(0, 1).text().replace(",", ".")) # eps
            case _:
                return

        # Расчет
        time_start = time()
        if self.glob_methods.currentText() == "Метод Монте-Карло":
            if self.loc_methods.currentText() == "Метод Нелдера-Мида":
                min_point, global_history, local_history = Optimizator.monte_karlo(
                    func, n_vars, lower_x, upper_x, N, Optimizator.nelder_mead,
                    eps_loc, N_loc
                )
            elif self.loc_methods.currentText() == "Метод Пауэлла":
                min_point, global_history, local_history = Optimizator.monte_karlo(
                    func, n_vars, lower_x, upper_x, N, Optimizator.powell,
                    eps_loc, N_loc
                )
            else:
                return
        elif self.glob_methods.currentText() == "Метод имитации отжига":
            if self.loc_methods.currentText() == "Метод Нелдера-Мида":
                min_point, global_history, local_history = Optimizator.annealing_imitation(
                    func, n_vars, lower_x, upper_x, Tmax, L, r, eps, Optimizator.nelder_mead,
                    eps_loc, N_loc
                )
            elif self.loc_methods.currentText() == "Метод Пауэлла":
                min_point, global_history, local_history = Optimizator.annealing_imitation(
                    func, n_vars, lower_x, upper_x, Tmax, L, r, eps, Optimizator.powell,
                    eps_loc, N_loc
                )
            else:
                return

        # Получаем результаты
        time_end = time() - time_start 
        min_value = func(min_point)
        vars = func.get_vars()

        # Вывод результатов
        self.result_time_label.setText(f"<b>Время работы алгоритма:</b> {round(time_end, 4)} сек")
        self.result_time_label.show()
        self.result_value_label.setText(f"<b>Минимум функции f(x*):</b> {round(min_value, 6)}")
        self.result_value_label.show()
        self.result_point_label.setText("<b>Минимум достигается в точке:</b>")
        self.result_point_label.show()
        self.result_point.setText(f"({'; '.join(vars)}) = ({'; '.join(map(lambda x: str(round(x, 6)), min_point))})")
        self.result_point.show()

        # Рисуем график
        # TODO : РИСОВАНИЕ ГРАФИКА

    