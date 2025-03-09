from collections import deque
import numpy as np
from sympy import Interval
from copy import deepcopy

from backend.Function import Function

class IntervalOptimizer:
    def __init__(self, eps: float):
        self.eps = eps

    def simple_optimize(self, func, n_vars: int, x_low: list, x_high: list) -> tuple:
        p = [[x_low[i], x_high[i]] for i in range(n_vars)]  # Начальный брус - заданные интервалы Х
        p_ = [Interval(*i) for i in p]  # Для вычислений
        f_ = func(p_)    # Вычисление естественной функции включения
        f_min = func(self.__mid(p))    # Для отбрасывания неподходящих брусов
        L = deque([(p, f_)])

        glob_history = [f_.start.evalf()]
        last_box = p
        while len(L) > 0 and (L[0][1].end.evalf() - L[0][1].start.evalf()) > self.eps:
        # while np.any(self.__wid(L[0][0]) > self.eps):
            print("ITERATION", len(glob_history) + 1, len(L))
            print("WID:", self.__wid(L[0][0]), L[0][0])
            
            current_box = L[0][0]

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
            funcs = sorted([(subbox1, f_1), (subbox2, f_2)], key=lambda x: x[1].start.evalf())
            print("FUNCTIONS", funcs)

            # Фильтруем и добавляем
            for f in funcs:
                if f[1].start.evalf() <= f_min and self.__monotonic_test(func, f[0]):
                    L.append(f)
            
            # Обновляем f_min, сохраняем историю
            if len(L) > 0:
                f_min = func(self.__mid(L[0][0]))
                # f_min = min(f_min, f_1.end, f_2.end)
                glob_history.append(L[0][1].start.evalf())  # для истории сохраняем ОЦЕНКИ ФУНКЦИИ, а не точки
                last_box = L[0][0]

        x_min = self.__mid(last_box)
        return x_min, glob_history


    def full_optimize(self, func, n_vars: int, x_low: list, x_high: list, eps: float) -> tuple:
        p = [[x_low[i], x_high[i]] for i in range(n_vars)]  # Начальный брус - заданные интервалы Х
        p_ = [Interval(*i) for i in p]  # Для вычислений
        f_ = func(p_)    # Вычисление естественной функции включения
        f_min_high = f_.end # Верхняя граница оценки минимума
        f_min_low = f_.start # Нижняя граница оценки минимума
        L = deque([(p, f_)])
        L_res = [] # Список глобальных минимумов

        flag = True
        while len(L) > 0 or flag:
            current_box = L[0][0]

            # Считаем точку разбиения
            bisection_index = np.argmax(self.__wid(current_box))
            bisection_interval = current_box[bisection_index]
            bisection_center = (bisection_interval[1] + bisection_interval[0]) / 2

            # Разделяем текущий брус на подбрусы
            subbox1, subbox2 = deepcopy(current_box), deepcopy(current_box)
            subbox1[bisection_index] = [subbox1[bisection_index][0], bisection_center]
            subbox2[bisection_index] = [bisection_center, subbox2[bisection_index][1]]

            # Оценки функции в новых брусах
            subbox1_ = [Interval(*i) for i in subbox1]
            subbox2_ = [Interval(*i) for i in subbox2]

            f_1 = func(subbox1_)
            f_2 = func(subbox2_)

            # Сортируем по возрастанию
            funcs = sorted([(subbox1, f_1), (subbox2, f_2)], key=lambda x: x[1].start)

            # Проверяем все необходимые тесты и добавляем брусы, прошедшие их
            for f in funcs:
                if self.__monotonic_test(func, f[0]) \
                    and self.__low_point_test(func, f[0], f_min_high) \
                    and self.__convexity_test(func, f[0]):
                        L.append(f)

            # f_min_high = min(f_min_high, f(self.__mid(L[0][0])))
            # f_min_low = max(f_min_low, f(self.__mid(L[0][0])))

            L_new = []
            for i in range(len(L)):
                if self.__middle_point_test(func, self.__mid(L[i][0]), f_min_low):
                    L_new.append(L[i])
            L = L_new

            # Обновление верхней и нижней оценки глобального минимума
            f_min_high = min(f_min_high, func(self.__mid(L[0][0])))
            f_min_low = L[0][0].start

            # Сохранение новых глобальных минимумов, если они нашлись
            if f_min_high - f_min_low < eps or self.__wid(L[0][0]) < eps:
                L_res.append(L[0][0])
            else:
                flag = False

            x_mins = [self.__mid(box) for box, f_box in L_res]
            print(x_mins)
            glob_history = []
            return x_mins[0], glob_history

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
    
    def __gradient_estimation(self, func, box: list) -> list[Interval]:
        """
        Возвращает естественную функцию включения для градиента функции.
        Возвращает список ИНТЕРВАЛОВ.
        """
        box_ = [Interval(*i) for i in box]
        gradient = func.get_gradient()  # Список градиентов соответственно переменным функции
        gradient_estimations = []   # Естественные функции включения по каждой производной
        for g in gradient:
            gf = Function(g, is_grad=True)
            gradient_estimations.append(gf(box_))
        return gradient_estimations
    
    def __hessian_estimation(self, func, box: list) -> list[Interval]:
        """
        Возвращает естественную функцию включения для матрицы Гессе функции.
        Возвращает список списков ИНТЕРВАЛОВ.
        """
        box_ = [Interval(*i) for i in box]
        hessian = func.get_hessian()  # Список списков производных (как в матрице Гессе)
        hessian_estimations = []   # Естественные функции включения для каждого элемента
        for row in hessian:
            hessian_estimations.append([])
            for i in range(len(row)):
                hf = Function(row[i], is_grad=True)
                hessian_estimations[i].append(hf(box_))
        return hessian_estimations

    def __centered_estimation(self, func, box: list) -> Interval:
        """
        Центрированная функция включения, возвращает интервал значений функции.
        """
        m = self.__mid(box)
        centered_diff = np.array(box) - np.array(m) # (x - m)
        gradient_estimations = self.__gradient_estimation(func, box)
        grad_center_mul = [0, 0]
        for i in range(len(centered_diff)): # [gT]([x]) * (x - m)
            mul = [centered_diff[i][0] * gradient_estimations[i].start.evalf(), 
                centered_diff[i][0] * gradient_estimations[i].end.evalf(),
                centered_diff[i][1] * gradient_estimations[i].start.evalf(), 
                centered_diff[i][1] * gradient_estimations[i].end.evalf()]
            grad_center_mul[0] += min(mul)
            grad_center_mul[1] += max(mul)
        f_m = func(m)   #  f(m), просто число
        result = np.array(grad_center_mul) + f_m    # f(m) + [gT]([x]) * (x - m), интервал
        return Interval(*result)

    def __middle_point_test(self, func, mid, f_min_low) -> bool:
        """
        Тест на значение в средней точке.
        mid - точка, относительно которой надо выполнить тест
        Если нижняя оценка функции включения на брусе больше среднего, 
        то этот брус считается неперспективным. (?)
        Возвращает True, если брус остается, False - если откидывается
        """
        # f_center_low = self.__centered_estimation(func, box)
        # return not (f_center_low.start > func(mid))
        return not (f_min_low > func(mid))
    
    def __low_point_test(self, func, box, f_min_high) -> bool:
        """
        Тест на значение в нижней границе.
        Если нижняя оценка центрированной функции включения на брусе больше верхней оценки
        глобального минимума, то этот брус считается неперспективным.
        Возвращает True, если брус остается, False - если откидывается
        """
        f_center_low = self.__centered_estimation(func, box)
        return not (f_center_low.start > f_min_high)

    def __monotonic_test(self, func, box: list) -> bool:
        """
        Тест на НУО, монотонность на указанном брусе.
        Вернет True, если на брусе ИМЕЕТСЯ точка экстремума,
        то есть функция НЕМОНОТОННА на брусе.
        """
        gradient_estimations = self.__gradient_estimation(func, box)
        flag = True
        for g in gradient_estimations:  # Если все компоненты содержат ноль, то норм
            if g.end.evalf() < 0 or g.start.evalf() > 0:
                flag = False
                break
        return flag

    def __convexity_test(self, func, box: list) -> bool:
        """
        Тест на ДУО, выпуклость.
        Возвращает True, если брус остается, False - если откидывается
        """
        hessian_estimations = self.__hessian_estimation(func, box)
        for i in range(len(hessian_estimations)):
            if hessian_estimations[i][i].end < 0:
                return False
        return True