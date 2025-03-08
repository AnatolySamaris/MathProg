from collections import deque
import numpy as np
from sympy import Interval
from copy import deepcopy

class IntervalOptimizer:
    def __init__(self, eps: float):
        self.eps = eps

    def simple_optimize(self, func, n_vars: int, x_low: list, x_high: list) -> tuple:
        p = [[x_low[i], x_high[i]] for i in range(n_vars)]  # Начальный брус - заданные интервалы Х
        p_ = [Interval(*i) for i in p]  # Для вычислений
        f_ = func(p_)    # Вычисление естественной функции включения
        f_min = f_.end    # Для отбрасывания неподходящих брусов
        L = deque([(p, f_)])

        glob_history = []
        last_box = p
        while len(L) > 0 and (L[0][1].end - L[0][1].start) > self.eps:
        # while np.any(self.__wid(L[0][0]) > self.eps):
            print("ITERATION", len(glob_history) + 1, len(L))
            print("WID:", self.__wid(L[0][0]), L[0][0])
            glob_history.append(L[0][1].start)  # для истории сохраняем ОЦЕНКИ ФУНКЦИИ, а не точки
            current_box = L[0][0]
            last_box = current_box
            
            # Считаем точку разбиения
            bisection_index = np.argmax(self.__wid(current_box))
            bisection_interval = current_box[bisection_index]
            bisection_center = (bisection_interval[1] + bisection_interval[0]) / 2
            print("BISECTION", bisection_index, bisection_interval, bisection_center)
            # Разделяем брусы
            subbox1, subbox2 = deepcopy(current_box), deepcopy(current_box)
            subbox1[bisection_index] = [subbox1[bisection_index][0], bisection_center]
            subbox2[bisection_index] = [bisection_center, subbox2[bisection_index][1]]

            # Оценки функции в новых брусах
            subbox1_ = [Interval(*i) for i in subbox1]
            subbox2_ = [Interval(*i) for i in subbox2]
            print("SUBBOXES", subbox1_, subbox2_)
            print()
            f_1 = func(subbox1_)
            f_2 = func(subbox2_)

            # Убираем первый элемент
            L.popleft()

            # Сортируем по возрастанию
            funcs = sorted([(subbox1, f_1), (subbox2, f_2)], key=lambda x: x[1].start)

            # Фильтруем и добавляем
            for f in funcs:
                if f[1].start <= f_min:
                    L.append(f)
            
            # Обновляем f_min
            f_min = min(f_min, f_1.end, f_2.end)

        x_min = self.__mid(last_box)
        return x_min, glob_history


    def full_optimize(self, func, n_vars: int, x_low: list, x_high: list, eps: float) -> tuple:
        pass

    def __wid(self, box: list) -> np.array:
        """
        Возвращает ширину бруса по каждой координате.
        """
        box = np.array(box)
        wid = box[:, 1] - box[:, 0]
        return wid

    def __mid(self, box: list) -> np.array:
        """
        Возвращает вектор-середину бруса.
        Брус - список кортежей, где каждый кортеж - интервал по i-й переменной
        (Например: [(-1, 1), (-2, 2), (-3, 3)] - брус из трех переменных)
        """
        box = np.array(box)
        return np.mean(box, axis=1)

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