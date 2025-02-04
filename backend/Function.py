from typing import List

class Function:
    """
    Класс для парсинга функции, записанной в строке в стиле Latex.
    При успешном парсинге создает рабочую функцию.
    В конструктор принимает строку с описанной функцией.
    При вызове требуется передать вектор значений [x1, x2, ..., xn].
    """
    
    __ACTIONS = {
        # '-': (4, lambda a: -a),
        '^': (3, lambda a, b: a ** b),
        '*': (2, lambda a, b: a * b),
        '/': (2, lambda a, b: a / b),
        '+': (1, lambda a, b: a + b),
        '-': (1, lambda a, b: a - b),
        #sin, cos, ...
    }

    def __init__(self, str_func: str):
        self.parsed_func = self.__parse_func(str_func)
    
    def __call__(self, x: List[float]):
        return self.__calculate_func(x)
    
    def __parse_func(self, str_func: str):
        pass

    def __calculate_func(self, x: List[float]):
        pass
