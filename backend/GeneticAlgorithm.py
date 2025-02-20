import numpy as np

class GeneticAlgorithm:
    def __init__(self, k0: int, h: float, n: int, eps: float, p: float):
        """
        k0 - размер исходной популяции
        h - ширина интервала для кодирования
        n - масимальное число поколений
        eps - точность для критерия останова
        p - вероятность мутации
        gene - номер гена, который менять при мутации
        """
        self.k0 = k0
        self.h = h
        self.n = n
        self.eps = eps
        self.p = p
        self.x_max_len = None
    
    def solve(self, f, n_vars: int, x_low: float, x_high: float, stopping_criteria: str) -> list:
        """
        Выполняет генетический алгоритм
        return: x_min, glob_history
        """
        # максимальная длина особи
        self.x_max_len = self.__find_max_len(x_low, x_high)
        # начальная популяция
        population = self.__create_population(x_low=x_low, x_high=x_high, n_vars=n_vars)

        # повторяем алгоритм до тех пор, пока не достигнем нужного числа поколений
        for _ in range(self.n):
            if _ > 0: population = new_population
            pairs = self.__selection(population=population, f=f, n_vars=n_vars, x_low=x_low, x_high=x_high)
            children = self.__crossingover(pairs=pairs, length=self.x_max_len)
            mutation_children = self.__mutation(children=children, length=self.x_max_len)
            new_population = self.__reduction(population=population, children=mutation_children, n_vars=n_vars, x_low=x_low)

            if stopping_criteria == 'one_generation':
                f_values = [f(x) for x in new_population]
                if max(f_values) - min(f_values) < self.eps:
                    break
            elif stopping_criteria == 'two_generations':
                f_max_old = max([f(x) for x in population])
                f_max_new = max([f(x) for x in new_population])
                if abs(f_max_new - f_max_old) < self.eps:
                    break

        # выбор лучшей особи в конечной популяции
        best_individual = max([f(x) for x in population])

        return best_individual

    def __create_population(self, x_low: float, x_high: float, n_vars: int) -> list:
        """
        Создает начальную популяцию.
        """
        start_population = np.random.uniform(low=x_low, high=x_high, size=(self.k0, n_vars))
        encode_population = [self.__encode_point(x, x_low, x_high) for x in start_population]
        return encode_population

    def __selection(self, population: list, f, n_vars: int, x_low: float) -> list:
        """
        Выполняет выбор пар для скрещивания.
        """
        intervals = [0]
        decode_population = [self.__decode_point(x, n_vars, x_low) for x in population]
        f_values = [f(x) for x in decode_population]
        f_max = max(f_values)
        f_sum = sum(f_values)
        for x in decode_population:
            p = (f_max - f(x) + 1) / ((self.k0 * (f_max + 1)) - f_sum)
            intervals.append(intervals[-1] + p)

        pairs = []
        for i in range(len(population)):
            if i % 2 == 0: pairs.append([])
            random_number = np.random.uniform(0, 1)
            for j in range(len(intervals) - 1):
                if random_number >= intervals[j] and random_number < intervals[j+1]:
                    pairs[j/2].append(population[j])
                    break
        return pairs

    def __crossingover(self, pairs: list, length: int, type: str) -> list:
        """
        Выполняет скрещивание и возвращает список новых особей.
        """
        children = []
        for i in range(len(pairs)):
            if type == 'single_pont':
                l = np.random.randint(1, length-1) 
                children.append(pairs[i][0][:l] + pairs[i][1][l:])
                children.append(pairs[i][1][:l] + pairs[i][0][l:])
            elif type == 'two_point':
                l1 = np.random.randint(1, length-1)
                l2 = np.random.randint(1, length-1)
                min_point = min([l1, l2])
                max_point = max([l1, l2])
                children.append(pairs[i][0][:min_point] + pairs[i][1][min_point : max_point] + pairs[i][0][max_point:])
                children.append(pairs[i][1][:min_point] + pairs[i][0][min_point : max_point] + pairs[i][1][max_point:])
            elif type == 'uniform':
                child1 = ''
                child2 = ''
                for bit in length:
                    bit_property_child1 = np.random.uniform(0, 1)
                    if bit_property_child1 < 0.5:
                        child1 += pairs[i][0][bit]
                    else:
                        child1 += pairs[i][1][bit]
                    bit_property_child2 = np.random.uniform(0, 1)
                    if bit_property_child2 < 0.5:
                        child2 += pairs[i][0][bit]
                    else:
                        child2 += pairs[i][1][bit]
                children.append(child1)
                children.append(child2)
        return children

    def __mutation(self, children: list, length: int) -> list:
        """
        Выполняет мутацию некоторых новых особей.
        """
        mutation_children = children.copy()
        for child in mutation_children:
            random_number = np.random.uniform(0, 1)
            if random_number <= self.p:
                gene = np.random.randint(0, length)
                child[gene] = 1 - child[gene]
        return mutation_children

    def __reduction(self, population: list, children: list, n_vars: int, x_low: float) -> list:
        """
        Выполняет редукцию и возвращает измененную популяцию.
        """
        new_population = population + children
        new_population.sort(key=lambda x: self.f(self.__decode_point(x, n_vars, x_low)), reverse=True)
        return new_population[:self.k0]

    def __dec2gray(self, dec):
        return dec ^ (dec >> 1)

    def __gray2dec(self, gray):
        dec = gray
        mask = dec >> 1
        while mask != 0:
            dec ^= mask
            mask >>= 1
        return dec

    def __encode_point(self, x_low, x_high, point: list):  # массив вещественных чисел
        intervals_point = []
        for i, x in enumerate(point):
            for hi in range(0, np.ceil((x_high[i] - x_low[i]) / self.h)):
                if (x_low + hi * self.h) <= x <= (x_low + (hi + 1) * self.h):
                    intervals_point.append(hi)
        encoded_point = ""
        for x in intervals_point:
            enc_x = bin(self.__dec2gray(x))[2:]
            encoded_point += "0" * (self.x_max_len - len(enc_x)) + enc_x
        return encoded_point

    def __decode_point(self, point, n_vars, x_low):
        len_point = len(point)
        split_step = len_point // n_vars
        point_intervals = []
        for i in range(0, len_point, split_step):
            point_intervals.append(
                point[i*split_step : (i+1)*split_step]
            )
        point_values = []
        for i, xi in enumerate(point_values):
            left_xi_value = x_low[i] + xi * self.h
            right_xi_value = x_low[i] + xi * (self.h + 1)
            point_values.append(
                (left_xi_value + right_xi_value) / 2
            )
        return point_values
    
    def __find_max_len(self, x_low, x_high) -> int:
        """
        Расчет максимальной длины числа в коде Грея
        """
        assert len(x_low) == len(x_high), "Ограничения заданы некорректно!"
        max_val = np.max([np.ceil((x_high[i] - x_low[i]) / self.h) 
                       for i in range(len(x_low))])
        max_val_gray = bin(self.__dec2gray(max_val))[2:]
        return len(max_val_gray)

