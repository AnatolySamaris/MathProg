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
        # f_min = func(self.__mid(p))    # Для отбрасывания неподходящих брусов
        # mf, cf = self.__min_box_value(func, p), func(self.__mid(p))
        # print(f"MIN: {mf}, CENTER: {cf}")
        # m = list(self.__mid(p))  # mi, pi
        pm = [0.5] * n_vars
        f_min = self.__min_box_value(func, p)
        L = deque([(p, f_)])

        cnt_removed_minf = 0
        cnt_removed_test = 0
        cnt_max_Llen = 0

        glob_history = [f_.start.evalf()]
        last_box = p
        while len(L) > 0 and (L[0][1].end.evalf() - L[0][1].start.evalf()) > self.eps:
        # while np.any(self.__wid(L[0][0]) > self.eps):
            print("ITERATION", len(glob_history) + 1, len(L))
            cnt_max_Llen = max(cnt_max_Llen, len(L))
            
            current_box = L[0][0]
            current_box_wid = self.__wid(current_box)
            print("BOX WID:", current_box_wid, current_box)

            # Считаем точку разбиения
            bisection_index = np.argmax(self.__wid(current_box))
            bisection_interval = current_box[bisection_index]

            print("BISECTION INDEX", bisection_index)
            print("PM LIST", pm)
            # print(m[bisection_index], pm[bisection_index], current_box_wid[bisection_index], current_box_wid[bisection_index] * pm[bisection_index])
            bisection_point = current_box[bisection_index][0] + pm[bisection_index] * current_box_wid[bisection_index]
            # bisection_center = (bisection_interval[1] + bisection_interval[0]) / 2

            print("BISECTION", bisection_index, bisection_interval, bisection_point)
            # Разделяем брусы
            subbox1, subbox2 = deepcopy(current_box), deepcopy(current_box)
            subbox1[bisection_index] = [subbox1[bisection_index][0], bisection_point]
            subbox2[bisection_index] = [bisection_point, subbox2[bisection_index][1]]


            # Оценки функции в новых брусах
            print("SUBBOXES", subbox1, subbox2)
            subbox1_ = [Interval(*i) for i in subbox1]
            subbox2_ = [Interval(*i) for i in subbox2]
            # print("SUBBOXES", subbox1_, subbox2_)
            f_1 = func(subbox1_)
            f_2 = func(subbox2_)

            # Пересчет смещения разбиения
            f1_val = f_1.start.evalf()
            f2_val = f_2.start.evalf()
            if f1_val < 0 and f2_val > 0:
                f1_val = abs(f1_val)
                f2_val = f2_val + abs(f1_val)
            elif f2_val < 0 and f1_val > 0:
                f2_val = abs(f2_val)
                f1_val = f1_val + abs(f2_val)
            else:
                f1_val = abs(f1_val)
                f2_val = abs(f2_val)
            # print("M VALUES", m)
            # m[bisection_index] = float(subbox1[bisection_index][0] + current_box_wid[bisection_index] * f_1.start.evalf() / f_2.start.evalf())
            m = subbox1[bisection_index][0] + (f1_val * current_box_wid[bisection_index]) / (f1_val + f2_val)

            print("M CALC", subbox1[bisection_index][0], f1_val * current_box_wid[bisection_index], (f1_val + f2_val), m)
            pm[bisection_index] = (m - current_box[bisection_index][0]) / current_box_wid[bisection_index]
            # print("CALCULATION", subbox1[bisection_index][0], current_box_wid[bisection_index], (f_1.start * current_box_wid[bisection_index]) / (f_1.start + f_2.start),
                #   subbox1[bisection_index][0] + (f_1.start * current_box_wid[bisection_index]) / (f_1.start + f_2.start))
            print("CALCULATION", m, current_box[bisection_index][0], current_box_wid[bisection_index], (m - current_box[bisection_index][0]) / current_box_wid[bisection_index])

            # Убираем первый элемент
            L.popleft()

            # Сортируем по возрастанию
            funcs = sorted([(subbox1, f_1), (subbox2, f_2)], key=lambda x: x[1].start.evalf())
            print("FUNCTIONS", funcs)
            # print()

            # Фильтруем и добавляем
            for f in funcs:
                if f[1].start.evalf() <= f_min and self.__monotonic_test(func, f[0]):
                    L.append(f)
                else:
                    if f[1].start.evalf() > f_min:
                        cnt_removed_minf += 1
                    else:
                        cnt_removed_test += 1
            
            # Обновляем f_min, сохраняем историю
            if len(L) > 0:
                # print("UPDATING", L[0][0])
                # mf, cf = self.__min_box_value(func, L[0][0]), func(self.__mid(L[0][0]))
                # print(f"MIN: {mf}, CENTER: {cf}")
                # f_min = min(mf, cf)
                # f_min = func(self.__mid(L[0][0]))
                # f_min = min(f_min, f_1.end, f_2.end)
                glob_history.append(L[0][1].start.evalf())  # для истории сохраняем ОЦЕНКИ ФУНКЦИИ, а не точки
                last_box = L[0][0]
                print(f"FUNC = {L[0][1]}; WIDTH {L[0][1].end.evalf() - L[0][1].start.evalf()}")
                print()

        x_min = self.__mid(last_box)
        print("=" * 20)
        print(f"REMOVED FUNC = {cnt_removed_minf}, TEST = {cnt_removed_test}; MAX L LEN = {cnt_max_Llen}")
        print("=" * 20)
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

    def __min_box_value(self, func, box) -> float:
        lower_x = []
        upper_x = []
        max_wid = None
        for x in box:
            lower_x.append(x[0])
            upper_x.append(x[1])
            max_wid = x[1]-x[0] if max_wid is None else max(max_wid, x[1] - x[0])
        N = int(max_wid / self.eps)
        
        # print(f"\tN points: {N}")
        glob_points = np.random.uniform(lower_x, upper_x, (N, len(lower_x)))
        glob_values = func(glob_points.T)
        # glob_values = np.array([func(point) for point in glob_points])
        x_min = glob_points[np.argmin(glob_values)]
        # print(f'\tARGMIN: {x_min}')

        # x_min = self.__monte_karlo(
        #     func, len(lower_x), lower_x, upper_x, N
        # )
        x_min = self.__gradient_descent(
            func, x_min, lower_x, upper_x, N, 0.1 * self.eps, 0.1 * self.eps, 0.01
        )
        min_box_value = func(x_min)
        return min_box_value

    def __middle_point_test(self, func, mid, box) -> bool:
        """
        Тест на значение в средней точке.
        mid - точка, относительно которой надо выполнить тест
        """
        pass

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

    def __convexity_test(self) -> bool:
        """
        Тест на ДУО, выпуклость.
        """
        pass

    def __monte_karlo(self, f, n_vars: int, x_low: list, x_high: list, N: int):
        x_low = np.array(x_low)
        x_high = np.array(x_high)

        x_min = np.random.uniform(low=x_low, high=x_high, size=n_vars)

        for _ in range(N):
            x = np.random.uniform(low=x_low, high=x_high, size=n_vars)
            if x_min[0] is None or f(x) < f(x_min):
                x_min = x

        return x_min

    def __gradient_descent(self, f, x_start, x_low: list, x_high: list,  N_loc: int, eps_loc: float, h: float, learning_rate: float):
        x_min = x_start.copy()

        def numerical_gradient(f, x, h):
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
            return grad
        
        for _ in range(N_loc):
            grad = numerical_gradient(f, x_min, h)
            # if _ == 0: print(grad)
            x = x_min - learning_rate * grad
            # if _ == 0: print(x)
            x = np.clip(x, x_low, x_high)
            # if _ == 0: print(x)
            if np.linalg.norm(f(x) - f(x_min)) < eps_loc:
                break
            x_min = x
        
        return x_min