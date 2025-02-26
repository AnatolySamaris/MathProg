import numpy as np
from scipy.optimize import minimize
from GeneticAlgorithm import GeneticAlgorithm

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
        x_low = np.array(x_low)
        x_high = np.array(x_high)

        x_min = np.random.uniform(low=x_low, high=x_high, size=n_vars)
        glob_history = [x_min.copy()]
        symmetry = True
        centers = (x_low + x_high) / 2

        for _ in range(N):
            x = np.random.uniform(low=x_low, high=x_high, size=n_vars)
            if x_min[0] is None or f(x) < f(x_min):
                x_min = x
                glob_history.append(x_min)
            if f(x) - f(2 * centers - x) > 1e-6: symmetry = False

        x_min, loc_history = loc_method(f, x_min, x_low, x_high, *args, **kwargs)
        if symmetry:
            if np.all(abs(x_min - centers) < 1): symmetry = None
            else: symmetry = 2 * centers - x_min
        else: symmetry = None
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
        x_low = np.array(x_low)
        x_high = np.array(x_high)

        x_min = np.random.uniform(low=x_low, high=x_high, size=n_vars)
        glob_history = [x_min.copy()]
        symmetry = True
        centers = (x_low + x_high) / 2

        while T > eps:
            for _ in range(L):
                x = x_min + np.random.uniform(low=-eps, high=eps, size=n_vars)
                x = np.clip(x, x_low, x_high)
                delta = f(x) - f(x_min)
                if delta <= 0 or np.exp(-delta / T) > np.random.uniform(low=0, high=1): 
                    x_min = x
                    glob_history.append(x_min)
                if f(x) - f(2 * centers - x) > 1e-6: symmetry = False
            T = r * T
        
        x_min, loc_history = loc_method(f, x_min, x_low, x_high, *args, **kwargs)
        if symmetry:
            if np.all(abs(x_min - centers) < 1): symmetry = None
            else: symmetry = 2 * centers - x_min
        else: symmetry = None
        return (x_min, glob_history, loc_history, symmetry)

    def genetic_algorithm(f, n_vars: int, x_low: list, x_high: list, k0: int, h: float, n: int, eps: float, p: float, loc_method, *args):
        ga = GeneticAlgorithm(k0, h, n, eps, p)
        x_min, glob_history = ga.solve(f, n_vars, x_low, x_high, 'one_generation', 'uniform')
        x_min, loc_history = loc_method(f, x_min, x_low, x_high, *args)
        return (x_min, glob_history, loc_history, None)
    

    def without_local_optimization(f, x_start, x_low: list, x_high: list, *args, **kwargs):
        loc_history = []
        return x_start, loc_history
    
    def nelder_mead(f, x_start, x_low: list, x_high: list, N_loc: int, eps_loc: float):
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
    
    def powell(f, x_start, x_low: list, x_high: list, N_loc: int, eps_loc: float):
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
    
    def bfgs(f, x_start, x_low: list, x_high: list, N_loc: int, eps_loc: float, h: float):
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

    def gradient_descent(f, x_start, x_low: list, x_high: list,  N_loc: int, eps_loc: float, h: float, learning_rate: float):
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
            if _ == 0: print(grad)
            x = x_min - learning_rate * grad
            if _ == 0: print(x)
            x = np.clip(x, x_low, x_high)
            if _ == 0: print(x)
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