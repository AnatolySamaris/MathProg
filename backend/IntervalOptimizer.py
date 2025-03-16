from collections import deque
import numpy as np
from sympy import Interval, FiniteSet
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
            print("ITERATION", len(glob_history), len(L))
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

            print("CALC F1", subbox1_, f_1)
            print("CALC F2", subbox2_, f_2)

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
            m = subbox1[bisection_index][0] + (f1_val * current_box_wid[bisection_index]) / (f1_val + f2_val + 1e-6)
            # min_delta = 0.1 * (subbox1[bisection_index][1] - subbox1[bisection_index][0])
            # m = np.clip(m, 
            #     subbox1[bisection_index][0] + min_delta, 
            #     subbox1[bisection_index][1] - min_delta
            # )

            print("M CALC", subbox1[bisection_index][0], f1_val * current_box_wid[bisection_index], (f1_val + f2_val), m)
            pm[bisection_index] = (m - current_box[bisection_index][0]) / current_box_wid[bisection_index]
            pm[bisection_index] = np.clip(pm[bisection_index], 0.1, 0.9)
            # print("CALCULATION", subbox1[bisection_index][0], current_box_wid[bisection_index], (f_1.start * current_box_wid[bisection_index]) / (f_1.start + f_2.start),
                #   subbox1[bisection_index][0] + (f_1.start * current_box_wid[bisection_index]) / (f_1.start + f_2.start))
            print("CALCULATION", m, current_box[bisection_index][0], current_box_wid[bisection_index], (m - current_box[bisection_index][0]) / current_box_wid[bisection_index])
            print("CLIPPED PM", np.clip(pm[bisection_index], 0.1, 0.9))

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
                        print(f"REMOVE FUNC: {f[1].start.evalf()} is upper than {f_min}")
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


    def full_optimize(self, func, n_vars: int, x_low: list, x_high: list, eps: float, n_mins: int) -> tuple:
        p = [[x_low[i], x_high[i]] for i in range(n_vars)]  # Начальный брус - заданные интервалы Х
        p_ = [Interval(*i) for i in p]  # Для вычислений
        f_ = func(p_)    # Вычисление естественной функции включения
        f_min_high = f_.end.evalf() # Верхняя граница оценки минимума
        f_min_low = f_.start.evalf() # Нижняя граница оценки минимума
        L = deque([(p, f_)])
        L_res = [] # Список глобальных минимумов
        glob_history = []

        flag = True
        count = 0
        w_eps = 1e-50
        while (len(L) > 1 or flag) and np.all(self.__wid(L[0][0]) > w_eps):
            count += 1
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
            print('SUBBOXES', subbox1, f_1, subbox2, f_2)

            m = self.__mid(L[0][0])
            f_low = L[0][1].start.evalf()

            # Убираем первый элемент
            L.popleft()

            # Сортируем по возрастанию
            funcs = sorted([(subbox1, f_1), (subbox2, f_2)], key=lambda x: x[1].start.evalf())

            # Проверяем все необходимые тесты и добавляем брусы, прошедшие их
            # and self.__middle_point_test(func, f_min_high, f[0]) \
            for f in funcs:
                if self.__monotonic_test(func, f[0]) \
                    and self.__low_point_test(func, f[0], f_min_high) \
                    and self.__convexity_test(func, f[0]):
                        L.append(f)
                        # print('APPEND', L)

            # по личной инициативе
            # конкретно этот код - совсем старый, надо смотреть ниже
            # f_min_high = min(f_min_high, f(self.__mid(L[0][0])))
            # f_min_low = max(f_min_low, f(self.__mid(L[0][0])))

            # Обновление верхней и нижней оценки глобального минимума
            # f_min_high = min(f_min_high, func(self.__mid(L[0][0])))
            # f_min_low = L[0][1].start.evalf()

            # это брать
            # L_new = deque()
            # for i in range(len(L)):
            #     if self.__middle_point_test(func, self.__mid(L[i][0]), f_min_low):
            #         # print('APPEND', L[i])
            #         L_new.append(L[i])
            # L = L_new

            # это брать
            # Обновление верхней и нижней оценки глобального минимума
            # (наверное, лучше здесь, а не выше, потому что изначально в f_min_low и так L[0][1].start.evalf())
            # f_min_high = min(f_min_high, func(self.__mid(L[0][0])))
            # f_min_low = L[0][1].start.evalf()
            # glob_history.append(f_min_high-f_min_low)

            
            # по лекции
            f_min_high = min(f_min_high, func(m))
            f_min_low = f_low
            glob_history.append(f_min_high-f_min_low)

            L_new = deque()
            for i in range(len(L)):
                if self.__middle_point_test(func, f_min_high, L[i][0]):
                    # print('APPEND', L[i])
                    L_new.append(L[i])
            L = L_new

            # print('FIRST', L[0][0])

            print('FIRST_INTERVAL', L[0][0])
            # print('WID', self.__wid(L[0][0]))
            # Сохранение новых глобальных минимумов, если они нашлись
            if f_min_high - f_min_low < eps or all(wid < eps for wid in self.__wid(L[0][0])):
                # print('YES')
                L_res.append(L[0])
            else:
                # print('NO')
                flag = False

        # x_mins = [self.__mid(box) for box, f_box in L_res]
        # x_mins_sort = sorted(x_mins, key=lambda x: self.__wid(x))
        # x_mins_result = x_mins_sort[:n_mins]
        # x_mins_sort = x_mins_sort[n_mins:]
        # for i in range(len(x_mins_result)):
        #     for j in range(i+1, len(x_mins_result)):
        #         if ((x_mins_result[i].start.evalt() >= x_mins_result[j].start.evalt() and \
        #             x_mins_result[i].end.evalt() <= x_mins_result[j].end.evalt()) or \
        #             (x_mins_result[j].start.evalt() >= x_mins_result[i].start.evalt() and \
        #             x_mins_result[j].end.evalt() <= x_mins_result[i].end.evalt())):
        #                 new_interval = Interval([
        #                     max(x_mins_result[i].start.evalt(), x_mins_result[j].start.evalt()),
        #                     min(x_mins_result[i].end.evalt(), x_mins_result[j].end.evalt()),
        #                 ])
        #                 x_mins_result = x_mins_result[:i] + x_mins_result[i+1 : j] + x_mins_result[j+1:]
        #                 x_mins_result.append(new_interval)

        # 
        p_mins = [el[0] for el in L_res]
        x_mins_sort = sorted(p_mins, key=lambda x: sum(self.__wid(x)))
        # print('X_MINS_SORT', x_mins_sort)
        # x_mins_sort = [[Interval(*bounds) for bounds in interval_pair] for interval_pair in x_mins_sort]

        # epsilon = 1e-2
        # for i in range(len(x_mins_sort)):
        #     if isinstance(x_mins_sort[i], FiniteSet):
        #         x_mins_sort[i] = Interval(float(interval_pair.value), float(interval_pair.value) + epsilon)
        
        # processed_x_mins_sort = []
        # for interval_list in x_mins_sort:
        #     print('YES')
        #     processed_interval_list = []
        #     for interval_pair in interval_list:
        #         if isinstance(interval_pair, FiniteSet):
        #             print('INTERVAL_PAIR_FINITE', interval_pair)
        #             print(Interval(float(interval_pair.args[0]), float(interval_pair.args[0]) + epsilon))
        #             processed_interval = Interval(float(interval_pair.args[0]), float(interval_pair.args[0]) + epsilon)
        #         else:
        #             processed_interval = interval_pair
        #         processed_interval_list.append(processed_interval)
        #     processed_x_mins_sort.append(processed_interval_list)
        # x_mins_sort = processed_x_mins_sort
        x_mins_result = []
        for interval_pair in x_mins_sort:
            if len(x_mins_result) >= n_mins:
                break
            # Проверяем пересечение с уже добавленными интервалами
            intersect = False
            for i, existing_interval_pair in enumerate(x_mins_result):
                # Проверяем пересечение по всем переменным
                new_interval_pair = []
                for interval, existing_interval in zip(interval_pair, existing_interval_pair):
                    print(interval, existing_interval)
                    intersection = self.__intersect_intervals(interval, existing_interval)
                    if intersection is None:
                        # Если хотя бы по одной переменной нет пересечения, выходим
                        break
                    new_interval_pair.append(intersection)
                else:
                    # Если пересечение найдено по всем переменным
                    x_mins_result.pop(i)
                    x_mins_result.append(new_interval_pair)
                    intersect = True
                    break
            # Если пересечений не было, просто добавляем интервал
            if not intersect:
                x_mins_result.append(interval_pair)

        print('X_MINS_RESULT', x_mins_result)
        x_mins = [self.__mid(box) for box in x_mins_result]
        print('X_MINS', x_mins)
        print('LEN_X_MINS', len(x_mins_result))
        print('COUNT', count)

        return x_mins[0], glob_history
    
    def __intersect_intervals(self, interval1, interval2):
        """
        Проверяет, пересекаются ли два интервала.
        Возвращает новый интервал как пересечение или None, если пересечения нет.
        """
        # eps = 0.01
        # if isinstance(interval1, FiniteSet):
        #     interval1_start = interval1.args[0]
        #     interval1_end = interval1.args[0] + eps
        # else:
        #     interval1_start = interval1.start.evalf()
        #     interval1_end = interval1.end.evalf()
        # if isinstance(interval2, FiniteSet):
        #     interval2_start = interval2.args[0]
        #     interval2_end = interval2.args[0] + eps
        # else:
        #     interval2_start = interval2.start.evalf()
        #     interval2_end = interval2.end.evalf()
        # new_start = max(interval1_start, interval2_start)
        # new_end = min(interval1_end, interval2_end)
        new_start = max(interval1[0], interval2[0])
        new_end = min(interval1[1], interval2[1])
        # if new_end == new_start: new_end += eps
        if new_start <= new_end:
            # return Interval(new_start, new_end)
            return [new_start, new_end]
        return None

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
        # print('GRADIENT')
        """
        Возвращает естественную функцию включения для градиента функции.
        Возвращает список ИНТЕРВАЛОВ.
        """
        box_ = [Interval(*i) for i in box]
        gradient = func.get_gradient()  # Список градиентов соответственно переменным функции
        # print('GRADIENT', gradient)
        gradient_estimations = []   # Естественные функции включения по каждой производной
        for g in gradient:
            gf = Function(str(g), is_grad=True)
            gradient_estimations.append(gf(box_))
        return gradient_estimations
    
    def __hessian_estimation(self, func, box: list) -> list[Interval]:
        """
        Возвращает естественную функцию включения для матрицы Гессе функции.
        Возвращает список списков ИНТЕРВАЛОВ.
        """
        # print('YES')
        box_ = [Interval(*i) for i in box]
        hessian = func.get_hessian()  # Список списков производных (как в матрице Гессе)
        # print('HESSIAN', hessian)
        hessian_estimations = []   # Естественные функции включения для каждого элемента
        for j in range(len(hessian)):
            hessian_estimations.append([])
            for i in range(len(hessian[j])):
                hf = Function(hessian[j][i], is_grad=True)
                hessian_estimations[j].append(hf(box_))
        # print('estimation', hessian_estimations)
        return hessian_estimations

    def __centered_estimation(self, func, box: list) -> Interval:
        """
        Центрированная функция включения, возвращает интервал значений функции.
        """
        m = self.__mid(box)
        centered_diff = np.array(box) - np.tile(np.array(m).reshape(-1, 1), 2) # (x - m)
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
    # def __middle_point_test(self, func, mid, f_min_low) -> bool:
        """
        Тест на значение в средней точке.
        mid - точка, относительно которой надо выполнить тест
        Если нижняя оценка функции включения на брусе больше среднего, 
        то этот брус считается неперспективным. (?)
        Возвращает True, если брус остается, False - если откидывается
        """
        f_center_low = self.__centered_estimation(func, box)
        return not (f_center_low.start.evalf() > mid)
        # return not (f_min_low > func(mid))
    
        # return not (func(box).start.evalt() > func(mid))
    
    def __low_point_test(self, func, box, f_min_high) -> bool:
        """
        Тест на значение в нижней границе.
        Если нижняя оценка центрированной функции включения на брусе больше верхней оценки
        глобального минимума, то этот брус считается неперспективным.
        Возвращает True, если брус остается, False - если откидывается
        """
        f_center_low = self.__centered_estimation(func, box)
        return not (f_center_low.start.evalf() > f_min_high)

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
        # print('HESSIAN_ESTIMATION', hessian_estimations)
        for i in range(len(hessian_estimations)):
            # print(hessian_estimations[i][i])
            if isinstance(hessian_estimations[i][i], Interval):
                hess_el = hessian_estimations[i][i].end.evalf()
            else:
                hess_el = hessian_estimations[i][i]
            if hess_el < 0:
                return False
        return True

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