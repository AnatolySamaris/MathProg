import numpy as np
from scipy.optimize import minimize

class Optimizator:
    def __init__(self):
        pass

    def monte_karlo(f, n_vars: int, x_low: list, x_high: list, N: int, loc_method, *args, **kwargs):
        """
        Ищет глобальный минимум функции методом Монте-Карло.
        В качестве параметров принимает:
            f - оптимизируемая функция
            n_vars - число переменных
            x_low, x_high - списки нижней и верхней границ x
            N - число итераций
            loc_method - метод локальной оптимизацииS
            *args, **kwargs - аргументы метода локальной оптимизации
        """
        x_min = np.random.uniform(low=x_low, high=x_high, size=n_vars)
        glob_history = [x_min.copy()]
        symmetry = False

        for _ in range(N):
            x = np.random.uniform(low=x_low, high=x_high, size=n_vars)
            if x_min[0] is None or f(x) < f(x_min):
                glob_history.append(x_min)
                x_min = x
            if f(x) == f(-x): symmetry = True

        x_min, loc_history = loc_method(f, x_min, x_low, x_high, *args, **kwargs)
        return (x_min, glob_history, loc_history, symmetry)
    
    def annealing_imitation(f, n_vars: int, x_low: list, x_high: list, T_max: float, L: int, r: float, eps: float, loc_method, *args, **kwargs):
        """
        Ищет глобальный минимум функции методом имитации отжига.
        В качестве параметров принимает:
            f - оптимизируемая функция
            n_vars - число переменных
            x_low, x_high - списки нижней и верхней границ x
            T_max - максимальное значение температуры
            L - число итераций для каждого T
            r - параметр для снижения T
            eps - малое вещественное число
            loc_method - метод локальной оптимизации
            *args, **kwargs - аргументы метода локальной оптимизации
        """
        T = T_max

        x_min = np.random.uniform(low=x_low, high=x_high, size=n_vars)
        glob_history = [x_min.copy()]
        symmetry = False

        while T > eps:
            for _ in range(L):
                x = x_min + np.random.uniform(low=-eps, high=eps, size=n_vars)
                x = np.clip(x, x_low, x_high)
                delta = f(x) - f(x_min)
                if delta <= 0 or np.exp(-delta / T) > np.random.uniform(low=0, high=1): 
                    x_min = x
                    glob_history.append(x_min)
                if f(x) == f(-x): symmetry = True
            T = r * T
        
        x_min, loc_history = loc_method(f, x_min, x_low, x_high, *args, **kwargs)
        return (x_min, glob_history, loc_history, symmetry)
    

    def nelder_mead(f, x_start, x_low: list, x_high: list, eps_loc: float, N_loc: int):
        """
        Ищет локальный минимум функции методом Нелдера-Мида.
        В качестве параметров принимает:
            f - оптимизируемая функция
            x_start - начальная точка
            x_low, x_high - списки нижней и верхней границ x
            eps_loc - значение для критерия останова
            N_loc - максимальное число итераций
        """
        loc_history = [x_start.copy()]
        def callback(x):
            x = np.clip(x, x_low, x_high)
            loc_history.append(x.copy())

        # bounds = [(x_low, x_high) for _ in range(len(x_start))]
        bounds = list(zip(x_low, x_high))
        
        x_min = minimize(
            f, 
            x_start, 
            method='nelder-mead', 
            bounds=bounds,
            options={'fatol': eps_loc, 'maxiter': N_loc},
            callback=callback
        )

        return (x_min.x, loc_history)
    
    def powell(f, x_start, x_low: list, x_high: list, eps_loc: float, N_loc: int):
        """
        Ищет локальный минимум функции методом Пауэлла.
        В качестве параметров принимает:
            f - оптимизируемая функция
            x_start - начальная точка
            x_low, x_high - списки нижней и верхней границ x
            eps_loc - значение для критерия останова
            N_loc - максимальное число итераций
        """
        loc_history = [x_start.copy()]
        def callback(x):
            x = np.clip(x, x_low, x_high)
            loc_history.append(x.copy())

        # bounds = [(x_low, x_high) for _ in range(len(x_start))]
        bounds = list(zip(x_low, x_high))
        
        x_min = minimize(
            f, 
            x_start, 
            method='powell', 
            bounds=bounds,
            options={'ftol': eps_loc, 'maxiter': N_loc},
            callback=callback
        )

        return (x_min.x, loc_history)
    
    def bfgs(f, x_start, x_low: list, x_high: list, eps_loc: float, N_loc: int, h: float):
        """
        Ищет локальный минимум функции методом L-BFGS-B (поддерживает границы).
        При этом якобиан считается численно.
        В качестве параметров принимает:
            f - оптимизируемая функция
            x_start - начальная точка
            x_low, x_high - списки нижней и верхней границ x
            eps_loc - значение для критерия останова
            N_loc - максимальное число итераций
            h - шаг для численного вычисления якобиана
        """
        loc_history = [x_start.copy()]
        def callback(x):
            loc_history.append(x.copy())

        bounds = list(zip(x_low, x_high))
        
        x_min = minimize(
            f, 
            x_start, 
            method='L-BFGS-B', 
            bounds=bounds,
            jac=None,
            options={'ftol': eps_loc, 'maxiter': N_loc, 'eps': h},
            callback=callback
        )

        return (x_min.x, loc_history)

    def gradient_descent(f, x_start, x_low: list, x_high: list, learning_rate: float, eps_loc: float, N_loc: int, h: float):
        """
        Реализует метод градиентного спуска с ограничениями.
        Параметры:
            f - целевая функция
            x_start - начальная точка
            x_low, x_high - списки нижней и верхней границ x
            learning_rate - скорость обучения
            N_loc - максимальное количество итераций
            eps_loc - критерий остановки (изменение нормы градиента)
            h - шаг для численного вычисления градиента
        """
        x_min = x_start.copy()
        loc_history = [x_min.copy()]

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
            x = x_min - learning_rate * grad
            x = np.clip(x, x_low, x_high)
            loc_history.append(x.copy())
            if np.linalg.norm(f(x) - f(x_min)) < eps_loc:
                break
            x_min = x
        
        return (x_min, loc_history)

    def tnc(f, x_start, x_low: list, x_high: list, eps_loc: float, h: float):
        """
        Ищет локальный минимум функции методом модифицированным методом Ньютона (учитывает границы
        и вместо гессиана использует метод сопряженных градиентов).
        В качестве параметров принимает:
            f - оптимизируемая функция
            x_start - начальная точка
            x_low, x_high - списки нижней и верхней границ x
            eps_loc - значение для критерия останова
            h - шаг для численного вычисления градиента
        """
        loc_history = [x_start.copy()]
        def callback(x):
            loc_history.append(x.copy())

        bounds = list(zip(x_low, x_high))
        
        x_min = minimize(
            f, 
            x_start, 
            method='TNC', 
            bounds=bounds,
            jac=None,
            options={'ftol': eps_loc, 'eps': h},
            callback=callback
        )

        return (x_min.x, loc_history)