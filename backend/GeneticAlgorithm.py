import numpy as np

class GeneticAlgorithm:
    def __init__(self, k0: int, h: float, n: int, eps: float, x_low: float, x_high: float, f):
        """
        k0 - размер исходной популяции
        h - ширина интервала для кодирования
        n - масимальное число поколений
        eps - точность для критерия останова
        """
        self.k0 = k0
        self.h = h
        self.n = n
        self.eps = eps
        self.x_low = x_low
        self.x_high = x_high
        self.f = f

    
    def solve(self, func) -> list:
        """
        return: x_min, glob_history
        """
        pass

    def __create_population(self):
        start_population = np.random.uniform(low=self.x_low, high=self.x_high, size=self.k0)
        encode_population = [self.__encode_point(x) for x in start_population]
        return encode_population

    def __selection(self, population):
        # properties = []
        intervals = [0]
        decode_population = [self.__decode_point(x) for x in population]
        f_arr = [self.f(x) for x in decode_population]
        f_max = max(f_arr)
        f_sum = sum(f_arr)
        for x in decode_population:
            p = (f_max - self.f(x) + 1) / ((self.k0 * (f_max + 1)) - f_sum)
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

    def __crossingover(self, pairs, length):
        children = []
        for i in range(len(pairs)): 
            l = np.random.randint(1, length-1) 
            # children.append([])
            # children[i].append(pairs[i][0][:l] + pairs[i][1][l:])
            # children[i].append(pairs[i][1][:l] + pairs[i][0][l:])
            children.append(pairs[i][0][:l] + pairs[i][1][l:])
            children.append(pairs[i][1][:l] + pairs[i][0][l:])
        return children

    def __mutation(self):
        pass

    def __reduction(self):
        pass 

    def __bit2grey(self):   # Анатолий
        pass 

    def __grey2bit(self):   # Толя
        pass

    def __encode_point(self):   # Толян
        pass

    def __decode_point(self):   # Толик
        pass