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
    
    def solve(self, f, n_vars: int, x_low: float, x_high: float, stopping_criteria: str) -> list:
        """
        Выполняет генетический алгоритм
        return: x_min, glob_history
        """
        # начальная популяция
        population = self.__create_population(x_low=x_low, x_high=x_high, n_vars=n_vars)
        # длина особи
        individual_length = len(population[0])

        # повторяем алгоритм до тех пор, пока не достигнем нужного числа поколений
        for _ in range(self.n):
            if _ > 0: population = new_population
            pairs = self.__selection(population=population, f=f)
            children = self.__crossingover(pairs=pairs, length=individual_length)
            mutation_children = self.__mutation(children=children, length=individual_length)
            new_population = self.__reduction(population=population, children=mutation_children)

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

    def __create_population(self, x_low, x_high, n_vars):
        """
        Создает начальную популяцию.
        """
        start_population = np.random.uniform(low=x_low, high=x_high, size=(self.k0, n_vars))
        encode_population = [self.__encode_point(x) for x in start_population]
        return encode_population

    def __selection(self, population, f):
        """
        Выполняет выбор пар для скрещивания.
        """
        intervals = [0]
        decode_population = [self.__decode_point(x) for x in population]
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

    def __crossingover(self, pairs, length, type):
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

    def __mutation(self, children, length):
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

    def __reduction(self, population, children):
        """
        Выполняет редукцию и возвращает измененную популяцию.
        """
        new_population = population + children
        new_population.sort(key=lambda x: self.f(self.__decode_point(x)), reverse=True)
        return new_population[:self.k0]

    def __bit2grey(self):   # Анатолий
        pass 

    def __grey2bit(self):   # Толя
        pass

    def __encode_point(self):   # Толян
        pass

    def __decode_point(self):   # Толик
        pass