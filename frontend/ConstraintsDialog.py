from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, 
    QLabel, QComboBox, QLineEdit, 
    QPushButton, QWidget
)
from PyQt5.QtGui import QFont, QRegularExpressionValidator
from PyQt5.QtCore import QRegularExpression

class ConstraintsDialog(QDialog):
    def __init__(self, variables, parent=None):
        super().__init__(parent)
        self.variables = variables
        self.constraints = {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Задать ограничения")
        self.layout = QVBoxLayout(self)

        self.font = QFont()
        self.font.setPointSize(10)

        regex = QRegularExpression(r"^[+-]?([0-9]*[.])?[0-9]+$")
        self.validator = QRegularExpressionValidator(regex)

        # Выпадающий список для выбора типа ограничений
        self.type_combo = QComboBox()
        self.type_combo.addItem("Задать для всех")
        self.type_combo.addItem("Задать для каждого")
        self.type_combo.setFixedHeight(40)
        self.type_combo.setFont(self.font)
        self.type_combo.currentIndexChanged.connect(self.update_ui)
        self.layout.addWidget(self.type_combo)

        # Контейнер для полей ввода
        self.input_container = QWidget()
        self.input_layout = QVBoxLayout(self.input_container)
        self.layout.addWidget(self.input_container)

        # Кнопки "Отмена" и "Принять"
        self.button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Отмена")
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

        self.accept_button = QPushButton("Принять")
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
        self.accept_button.clicked.connect(self.accept)
        self.accept_button.setEnabled(False)

        self.button_layout.addWidget(self.cancel_button)
        self.button_layout.addWidget(self.accept_button)
        self.layout.addLayout(self.button_layout)

        self.update_ui()

    def update_ui(self):
        while self.input_layout.count():
            item = self.input_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
            elif item.layout():
                while item.layout().count():
                    sub_item = item.layout().takeAt(0)
                    if sub_item.widget():
                        sub_item.widget().setParent(None)

        if self.type_combo.currentText() == "Задать для всех":
            row_layout = QHBoxLayout()
            self.lower_bound_input = QLineEdit()
            self.lower_bound_input.setFont(self.font)
            self.lower_bound_input.setValidator(self.validator)
            row_layout.addWidget(self.lower_bound_input)
            label_var = QLabel("<= xi <=")
            label_var.setFont(self.font)
            row_layout.addWidget(label_var)
            self.upper_bound_input = QLineEdit()
            self.upper_bound_input.setFont(self.font)
            self.upper_bound_input.setValidator(self.validator)
            row_layout.addWidget(self.upper_bound_input)
            self.input_layout.addLayout(row_layout)
            self.lower_bound_input.textChanged.connect(self.check_inputs)
            self.upper_bound_input.textChanged.connect(self.check_inputs)

        else:
            for var in self.variables:
                row_layout = QHBoxLayout()
                lower_bound_input = QLineEdit()
                lower_bound_input.setFont(self.font)
                lower_bound_input.setValidator(self.validator)
                row_layout.addWidget(lower_bound_input)
                label_var = QLabel(f"<= {var} <=")
                label_var.setFont(self.font)
                row_layout.addWidget(label_var)
                upper_bound_input = QLineEdit()
                upper_bound_input.setFont(self.font)
                upper_bound_input.setValidator(self.validator)
                row_layout.addWidget(upper_bound_input)
                self.input_layout.addLayout(row_layout)
                lower_bound_input.textChanged.connect(self.check_inputs)
                upper_bound_input.textChanged.connect(self.check_inputs)

    def check_inputs(self):
        if self.type_combo.currentText() == "Задать для всех":
            lower = self.lower_bound_input.text().strip()
            upper = self.upper_bound_input.text().strip()
            if lower and upper:
                self.accept_button.setEnabled(True)
            else:
                self.accept_button.setEnabled(False)
        else:
            all_filled = True
            for i in range(self.input_layout.count()):
                row_layout = self.input_layout.itemAt(i).layout()
                if row_layout:
                    lower = row_layout.itemAt(0).widget().text().strip()
                    upper = row_layout.itemAt(2).widget().text().strip()
                    if not lower or not upper:
                        all_filled = False
                        break
            self.accept_button.setEnabled(all_filled)

    def accept(self):
        self.constraints = {}
        if self.type_combo.currentText() == "Задать для всех":
            lower = float(self.lower_bound_input.text().strip())
            upper = float(self.upper_bound_input.text().strip())
            for var in self.variables:
                self.constraints[var] = [lower, upper]
        else:
            for i in range(self.input_layout.count()):
                row_layout = self.input_layout.itemAt(i).layout()
                if row_layout:
                    var = row_layout.itemAt(1).widget().text().split()[1]
                    lower = float(row_layout.itemAt(0).widget().text().strip())
                    upper = float(row_layout.itemAt(2).widget().text().strip())
                    self.constraints[var] = [lower, upper]

        super().accept()