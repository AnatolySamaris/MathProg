import numpy as np

class Optimizator:
    def __init__(self):
        pass

    def monte_karlo(f, n_vars, x_low: int, x_high: int, N: int, loc_method, *args, **kwargs):
        """
        Ищет глобальный минимум функции методом Монте-Карло.
        В качестве параметров принимает:
            f - оптимизируемая функция
            n_vars - число переменных
            x_low, x_high - нижняя и верхняя границы x
            N - число итераций
            loc_method - метод локальной оптимизацииS
            *args, **kwargs - аргументы метода локальной оптимизации
        """
        x_min = [None] * n_vars
        glob_history = []

        for _ in range(N):
            x = np.random.uniform(low=x_low, high=x_high, size=n_vars)
            if f(x) < f(x_min):
                glob_history.append(x_min)
                x_min = x

        x_min, loc_history = loc_method(x_min, *args, **kwargs)
        return (x_min, glob_history, loc_history)
    
    def annealing_imitation(f, n_vars, x_low: int, x_high: int, T_max: float, L: int, r: float, eps: float, loc_method, *args, **kwargs):
        """
        Ищет глобальный минимум функции методом имитации отжига.
        В качестве параметров принимает:
            f - оптимизируемая функция
            n_vars - число переменных
            x_low, x_high - нижняя и верхняя границы x
            T_max - максимальное значение температуры
            L - число итераций для каждого T
            r - параметр для снижения T
            eps - малое вещественное число
            loc_method - метод локальной оптимизацииS
            *args, **kwargs - аргументы метода локальной оптимизации
        """
        T = T_max
        glob_history = []

        x_min = np.random.uniform(low=x_low, high=x_high, size=n_vars)
        while T > eps:
            for _ in range(L):
                x = x_min + np.random.uniform(low=-eps, high=eps, size=n_vars)
                x = np.clip(x, x_low, x_high)
                delta = f(x) - f(x_min)
                if delta <= 0 or np.exp(-delta / T) > np.random.uniform(low=0, high=1): 
                    x_min = x
                    glob_history.append(x_min)
            T = r * T
        
        x_min, loc_history = loc_method(x_min, *args, **kwargs)
        return (x_min, glob_history, loc_history)