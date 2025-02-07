from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDesktopWidget, QFrame, QLineEdit, QComboBox, QTableWidget, QTableWidgetItem, QSizePolicy, QHeaderView
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from functools import partial
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp
from PyQt5.QtWidgets import QItemDelegate, QStyledItemDelegate

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

        # строка "Сообщение об ошибке" красным цветом
        self.error_message_label = QLabel("Сообщение об ошибке")
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

        # добавляем нижнюю часть в левую часть
        left_layout.addWidget(bottom_left_widget, stretch=65)

        # добавляем левую часть в главный layout
        main_layout.addWidget(left_widget, stretch=50)

        # разделитель между левой и правой частью
        main_separator = QFrame()
        main_separator.setFrameShape(QFrame.VLine)  # Вертикальная линия
        main_layout.addWidget(main_separator)

        # правая часть
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # верхняя часть правой части
        top_right_widget = QWidget()
        right_layout.addWidget(top_right_widget, stretch=40)

        # разделитель между верхней и нижней частью правой стороны
        right_separator1 = QFrame()
        right_separator1.setFrameShape(QFrame.HLine)  # Горизонтальная линия
        right_layout.addWidget(right_separator1)

        # нижняя часть правой части
        bottom_right_widget = QWidget()
        right_layout.addWidget(bottom_right_widget, stretch=60)

        # добавляем правую часть в главный layout
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