import numpy as np

class IntervalOptimizer:
    def __init__(self, eps: float):
        self.eps = eps

    def simple_optimize(self, func, n_vars: int, x_low: list, x_high: list) -> tuple:
        pass

    def full_optimize(self, func, n_vars: int, x_low: list, x_high: list, eps: float) -> tuple:
        pass

    def __mid(self, box: list) -> list:
        """
        Возвращает вектор-середину бруса.
        Брус - список кортежей, где каждый кортеж - интервал по i-й переменной
        (Например: [(-1, 1), (-2, 2), (-3, 3)] - брус из трех переменных)
        """
        box = np.array(box)
        return np.mean(box, axis=1)

    def __estimate_function(self, func, x_low, x_high) -> tuple:
        """
        Оценка функции включения, возвращает интервал значений функции.
        """
        num_points = int(1 / self.eps)
        points = np.column_stack([np.linspace(low, high, num_points) 
                                for low, high in zip(x_low, x_high)])
        func_values = func(points)
        min_value = func_values.min()
        max_value = func_values.max()
        return (min_value, max_value)

    def __centered_estimation(self, func, x_low, x_high) -> tuple:
        """
        Центрированная функция включения, возвращает интервал значений функции.
        """
        pass

    def __middle_point_test(self, func, box) -> bool:
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