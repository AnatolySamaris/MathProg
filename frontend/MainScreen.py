from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QDialog, QTextEdit,
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

from frontend.ConstraintsDialog import ConstraintsDialog

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

        self.constraints = {}

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
        self.function_input.setAlignment(Qt.AlignCenter)
        self.function_input.setValidator(function_validator)
        self.function_input.textChanged.connect(self.update_beautiful_function)
        function_input_layout.addWidget(fx_label)
        function_input_layout.addWidget(self.function_input)
        top_left_layout.addLayout(function_input_layout)

        # ОГРАНИЧЕНИЯ
        constraint_label = QLabel("<p>Ограничение для x<sub>i</sub></p>")
        constraint_label.setFont(font)
        top_left_layout.addWidget(constraint_label)

        # Кнопка "Задать ограничения"
        self.set_constraints_button = QPushButton("Задать ограничения")
        self.set_constraints_button.setFont(font)
        self.set_constraints_button.setMinimumHeight(32)
        self.set_constraints_button.setStyleSheet("""
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
        self.set_constraints_button.clicked.connect(self.open_constraints_dialog)
        top_left_layout.addWidget(self.set_constraints_button)

        # Поле для отображения ограничений
        self.constraints_display = QTextEdit()
        self.constraints_display.setFont(font)
        self.constraints_display.setAlignment(Qt.AlignCenter)
        self.constraints_display.setReadOnly(True)
        top_left_layout.addWidget(self.constraints_display)

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

        # Виджет для графика
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)

        # Создаем фигуру и холст для графика
        self.graph_figure = Figure()
        self.graph_canvas = FigureCanvas(self.graph_figure)

        # Добавляем холст в layout
        graph_layout.addWidget(self.graph_canvas)

        # Добавляем виджет с графиком в правую нижнюю часть
        bottom_right_layout.addWidget(graph_widget)

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

    def open_constraints_dialog(self):
        # Получаем переменные из функции
        try:
            func = Function(self.function_input.text())
            variables = func.get_vars()
        except Exception:
            self.error_message_label.setText("Сначала введите функцию!")
            return

        # Открываем диалоговое окно для задания ограничений
        dialog = ConstraintsDialog(variables, self)
        if dialog.exec_() == QDialog.Accepted:
            self.constraints = dialog.constraints
            # Отображаем ограничения в текстовом поле
            constraints_text = ", ".join([f"{var}: [{bounds[0]}, {bounds[1]}]" for var, bounds in self.constraints.items()])
            self.constraints_display.setText(constraints_text)

    # общие настройки таблиц
    def set_table_parameters(self, table, headers):
        # убираем индексы строк
        table.verticalHeader().setVisible(False)

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
        # очищаем содержимое таблицы
        table.clearContents()

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

        elif selected_method == "Метод Ньютона":
            headers = ["ε", "h"]
            self.set_table_parameters(table, headers)

        elif selected_method == "BFGS":
            headers = ["N", "ε", "h"]
            self.set_table_parameters(table, headers)

        elif selected_method == "Градиентный спуск":
            headers = ["N", "ε", "h", "λ"]
            self.set_table_parameters(table, headers)

        # выравнивание столбцов на всю ширину
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        # растягиваем таблицу на всю высоту
        table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

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
            self.error_message_label.setText("")
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
        if len(self.constraints_display.toPlainText().strip()) == 0:
            self.error_message_label.setText("Ограничения на переменные не заданы!")
            return False
        for col in range(self.glob_table.columnCount()):
            if not self.glob_table.item(0, col) or (self.glob_table.item(0, col).text().strip()) == 0:
                self.error_message_label.setText("Глобальный метод не инициализирован!")
                return False
        for col in range(self.loc_table.columnCount()):
            if not self.loc_table.item(0, col) or (self.loc_table.item(0, col).text().strip()) == 0:
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
        lower_x = []
        upper_x = []
        for x in func.get_vars():
            lower_x.append(self.constraints[x][0])
            upper_x.append(self.constraints[x][1])

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
            case "Метод Ньютона":
                eps_loc = float(self.loc_table.item(0, 0).text().replace(",", ".")) # eps
                h = float(self.loc_table.item(0, 1).text().replace(",", ".")) # h
            case "BFGS":
                N_loc = int(self.loc_table.item(0, 0).text())  # N
                eps_loc = float(self.loc_table.item(0, 1).text().replace(",", ".")) # eps
                h = float(self.loc_table.item(0, 2).text().replace(",", ".")) # h
            case "Градиентный спуск":
                N_loc = int(self.loc_table.item(0, 0).text())  # N
                eps_loc = float(self.loc_table.item(0, 1).text().replace(",", ".")) # eps
                h = float(self.loc_table.item(0, 2).text().replace(",", ".")) # h
                lr = float(self.loc_table.item(0, 3).text().replace(",", ".")) # lr
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
            elif self.loc_methods.currentText() == "Метод Ньютона":
                min_point, global_history, local_history = Optimizator.monte_karlo(
                    func, n_vars, lower_x, upper_x, N, Optimizator.tnc,
                    eps_loc, h
                )
            elif self.loc_methods.currentText() == "BFGS":
                min_point, global_history, local_history = Optimizator.monte_karlo(
                    func, n_vars, lower_x, upper_x, N, Optimizator.bfgs,
                    eps_loc, N_loc, h
                )
            elif self.loc_methods.currentText() == "Градиентный спуск":
                min_point, global_history, local_history = Optimizator.monte_karlo(
                    func, n_vars, lower_x, upper_x, N, Optimizator.gradient_descent,
                    lr, eps_loc, N_loc, h
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
            elif self.loc_methods.currentText() == "Метод Ньютона":
                min_point, global_history, local_history = Optimizator.annealing_imitation(
                    func, n_vars, lower_x, upper_x, Tmax, L, r, eps, Optimizator.tnc,
                    eps_loc, h
                )
            elif self.loc_methods.currentText() == "BFGS":
                min_point, global_history, local_history = Optimizator.annealing_imitation(
                    func, n_vars, lower_x, upper_x, Tmax, L, r, eps, Optimizator.bfgs,
                    eps_loc, N_loc, h
                )
            elif self.loc_methods.currentText() == "Градиентный спуск":
                min_point, global_history, local_history = Optimizator.annealing_imitation(
                    func, n_vars, lower_x, upper_x, Tmax, L, r, eps, Optimizator.gradient_descent,
                    lr, eps_loc, N_loc, h
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
        global_history_length = len(global_history)
        local_history_length = len(local_history)
        global_history_f = [func(x) for x in global_history]
        global_steps = list(range(0, global_history_length))
        local_history_f = [func(x) for x in local_history]
        local_steps = list(range(global_history_length - 1, global_history_length - 1 + local_history_length))
        self.graph_figure.clear()
        ax = self.graph_figure.add_subplot(111)
        ax.plot(global_steps, global_history_f, label="Глобальная оптимизация")
        ax.plot(local_steps, local_history_f, label="Локальная оптимизация")
        ax.set_xlabel("Итерации")
        ax.set_ylabel("Значение функции")
        ax.legend()
        ax.grid(True)
        self.graph_canvas.draw()