import numpy as np

class IntervalOptimizer:
    def __init__(self):
        pass

    def simple_optimize(self, func, n_vars: int, x_low: list, x_high: list, eps: float) -> tuple:
        pass

    def full_optimize(self, func, n_vars: int, x_low: list, x_high: list, eps: float) -> tuple:
        pass

    def __estimate_function(self, func, x_low, x_high) -> tuple:
        """
        Оценка функции включения, возвращает интервал значений функции.
        """
        pass

    def __centered_estimation(self, func, x_low, x_high) -> tuple:
        """
        Центрированная функция включения, возвращает интервал значений функции.
        """
        pass

    def __middle_point_test(self) -> bool:
        """
        Тест на значение в средней точке.
        """
        pass

    def __monotonic_test(self) -> bool:
        """
        Тест на НУО, монотонность.
        """
        pass

    def __convexity_test(self) -> bool:
        """
        Тест на ДУО, выпуклость.
        """
        pass