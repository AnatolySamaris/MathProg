from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QTabWidget, QTextEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class HelpWindow(QMainWindow):
    """
    Окно справки, которое содержит инструкции для пользователя по работе с программой.
    Справка разделена на блоки, параметры каждого из которых задаются в методе __init__.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справка")
        self.setFixedSize(800, 500)
        self.center()

        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(10)

        """
        Поля класса, содержащие текст каждого блока справки.
        """
        self.text_general = QTextEdit("""
            Программа позволяет пользователю исследовать различные методы глобальной оптимизации.<br>
                                      
            Интерфейс разделен на 4 части. Ниже представлено их более подробное описание.<br><br>
                                      
                1. Левая верхняя часть.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;1.1. Ввод функции.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;Можно вводить любые переменные или операторы в формате Latex.
                Вводимая функция отображается в верхней правой части. В случае ошибки при вводе функции появится сообщение 
                о неверном формате ввода.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;1.2. Ввод ограничений.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;Ограничения можно задать по кнопке "Задать ограничения" двумя способами: для всех переменных сразу
                или для каждой переменной отдельно.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;1.3. Сид генерации.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;Сид генерации позволяет задать начальное значение для генератора псевдо-случайных чисел.<br><br>
                                      
                2. Левая нижняя часть.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;2.1. Метод глобальной оптимизации.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;Можно выбрать любой метод глобальной оптимизации из выпадающего списка, а затем задать его
                параметры в таблице ниже.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;2.2. Метод локальной оптимизации.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;Для уточнения результата, полученного методом глобальной оптимизации, можно аналогичным образом
                выбрать метод локальной оптимизации из выпадающего списка и задать необходимые параметры
                в таблице.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;2.3. Кнопка "Рассчитать".<br>
                &nbsp;&nbsp;&nbsp;&nbsp;При нажатии на эту кнопку начинается выполнение вычислений. В случае, если начальные параметры
                были введены неверно, появится сообщение об ошибке.<br><br>
                                      
                3. Правая верхняя часть.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;Включает в себя отображение введенной функции и выходные параметры.<br><br>
                                      
                4. Правая нижняя часть.<br>
                &nbsp;&nbsp;&nbsp;&nbsp;Используется для отображения графика зависимости значения оптимизируемой функции от
                количества итераций.
        """)
        self.global_optimization = QTextEdit("""
            В программе реализованы следующие методы глобальной оптимизации:
            <ol>
                <li>Метод Монте-Карло</li>
                Входные параметры:
                <ul>
                    <li> N &mdash; количество итераций. <\li>
                </ul>
                                             
                <li>Метод имитации отжига</li>
                Входные параметры:
                <ul>
                    <li> T<sub>max</sub> &mdash; максимальная температура; <\li>
                    <li> L &mdash; число итераций для каждого T; <\li>
                    <li> r &mdash; параметр для снижения T; <\li>
                    <li> &epsilon; &mdash; малое вещественное число. <\li>
                </ul>
                                             
                <li>Генетический алгоритм</li>
                Входные параметры:
                <ul>
                    <li> k &mdash; размер начальной популяции; <\li>
                    <li> h &mdash; ширина интервала для кодирования вещественных чисел; <\li>
                    <li> N &mdash; максимальное количество поколений; <\li>
                    <li> &epsilon; &mdash; точность; <\li>
                    <li> p &mdash; вероятность мутации. <\li>
                </ul>
            </ol>
        """)
        self.local_optimization = QTextEdit("""
            После применения методов глобальной оптимизации пользователь может применить один из
            реализованных в программе методов локальной оптимизации для уточнения полученной точки, либо
            выбрать вариант "Не использовать локальный метод", и тогда поиск решения завершится после
            выполнения глобальной оптимизации.<br>
            
            Методы локальной оптимизации, реализованные в программе:
            <ol>
                <li>Метод Нелдера-Мида</li>
                Входные параметры:
                <ul>
                    <li> N &mdash; максимальное количество итераций; <\li>
                    <li> &epsilon; &mdash; точность. <\li>
                </ul>
                                             
                <li>Метод Пауэлла</li>
                Входные параметры:
                <ul>
                    <li> N &mdash; максимальное количество итераций; <\li>
                    <li> &epsilon; &mdash; точность. <\li>
                </ul>
                                             
                <li>Градиентный спуск</li>
                Входные параметры:
                <ul>
                    <li> N &mdash; максимальное количество итераций; <\li>
                    <li> &epsilon; &mdash; точность. <\li>
                    <li> h &mdash; шаг для вычисления градиента; <\li>
                    <li> &lambda; &mdash; скорость обучения; <\li>
                </ul>
                                            
                <li>Метод Ньютона</li>
                Входные параметры:
                <ul>
                    <li> &epsilon; &mdash; точность. <\li>
                    <li> h &mdash; шаг для вычисления градиента; <\li>
                </ul>
                                            
                <li>BFGS</li>
                Входные параметры:
                <ul>
                    <li> N &mdash; максимальное количество итераций; <\li>
                    <li> &epsilon; &mdash; точность. <\li>
                    <li> h &mdash; шаг для вычисления градиента; <\li>
                </ul>
            </ol>
        """)
        self.text_about = QTextEdit("""
            Математическое программирование.<br>
            &#169; ЛГТУ, 2025 г <br>
            Седых О.М., Целищев А.Е.
        """)

        """
        Установление шрифта font тексту каждого блока.
        """
        self.text_general.setFont(font)
        self.global_optimization.setFont(font)
        self.local_optimization.setFont(font)
        self.text_about.setFont(font)

        """
        Формирование виджета для отображения инструкций.
        """
        self.tab_widget = QTabWidget(self)
        self.tab_widget.addTab((self.text_general), "Общие сведения")
        self.tab_widget.addTab((self.global_optimization), "Методы глобальной оптимизации")
        self.tab_widget.addTab((self.local_optimization), "Методы локальной оптимизации")
        self.tab_widget.addTab((self.text_about), "О программе")

        """
        Запрет редактирования текста в каждом блоке инструкций.
        """
        for index in range(self.tab_widget.count()):
            self.text_edit = self.tab_widget.widget(index)
            self.text_edit.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        self.setCentralWidget(self.tab_widget)
        self.show()

    def center(self):
        """
        Располагает окно справки по центру экрана.
        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
