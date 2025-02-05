from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDesktopWidget, QFrame

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
        left_layout.addWidget(top_left_widget, stretch=35)

        # разделитель между верхней и нижней частью левой стороны
        left_separator1 = QFrame()
        left_separator1.setFrameShape(QFrame.HLine)  # Горизонтальная линия
        # left_separator1.setFrameShadow(QFrame.Sunken)
        left_layout.addWidget(left_separator1)

        # нижняя часть левой стороны (остальное)
        bottom_left_widget = QWidget()
        left_layout.addWidget(bottom_left_widget, stretch=65)

        # добавляем левую часть в главный layout
        main_layout.addWidget(left_widget, stretch=50)

        # разделитель между левой и правой частью
        main_separator = QFrame()
        main_separator.setFrameShape(QFrame.VLine)  # Вертикальная линия
        # main_separator.setFrameShadow(QFrame.Sunken)
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
        # right_separator1.setFrameShadow(QFrame.Sunken)
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