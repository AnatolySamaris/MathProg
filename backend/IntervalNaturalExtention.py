import numpy as np
import sympy as sp
from sympy import Interval, EmptySet
from sympy.core import numbers
from sympy.sets import FiniteSet
from fractions import Fraction

class IntervalNaturalExtention:
    def __init__(self):
        self.inf = float("inf")

    def eval_interval_expr(self, expr, interval_dict):
        """
        Рекурсивно вычисляет выражение, где переменные заменены на интервалы.
        :param expr: SymPy-выражение.
        :param interval_dict: Словарь, где ключи — переменные, а значения — интервалы или числа.
        :return: Результат вычисления (интервал или число).
        """
        # print('type_expr', type(expr), 'expr', expr)
        if isinstance(expr, sp.Symbol):
            # Если это переменная, заменяем её на интервал из словаря
            return interval_dict.get(expr, expr)  # Если переменной нет в словаре, возвращаем её как есть

        elif isinstance(expr, numbers.Half):    # Если это "Половина", 1/2
            return expr

        elif isinstance(expr, FiniteSet):   # Если это вырожденный интервал
            return expr.args[0]

        elif isinstance(expr, sp.Add):  # Обработка сложения
            intervals = [self.eval_interval_expr(arg, interval_dict) for arg in expr.args if arg != EmptySet]
            return self.__interval_add(*intervals)

        elif isinstance(expr, sp.Mul):  # Обработка умножения
            intervals = [self.eval_interval_expr(arg, interval_dict) for arg in expr.args if arg != EmptySet]
            return self.__interval_mul(*intervals)

        elif isinstance(expr, sp.Pow):  # Обработка степени
            base = self.eval_interval_expr(expr.base, interval_dict)
            exponent = self.eval_interval_expr(expr.exp, interval_dict)
            return self.__interval_pow(base, exponent)

        elif isinstance(expr, sp.sin):  # Обработка синуса
            arg = self.eval_interval_expr(expr.args[0], interval_dict)
            return self.__interval_sin(arg)

        elif isinstance(expr, sp.cos):  # Обработка косинуса
            arg = self.eval_interval_expr(expr.args[0], interval_dict)
            return self.__interval_cos(arg)

        elif isinstance(expr, sp.exp):  # Обработка экспоненты
            arg = self.eval_interval_expr(expr.args[0], interval_dict)
            return self.__interval_exp(arg)

        else:   # Если это число, возвращаем как есть
            return expr


    def __interval_add(self, *intervals):
        """
        Сложение интервалов.
        """
        intervals = [i for i in intervals if i != EmptySet]
        # for i in intervals:
            # print("\tFROM SUM", type(i), i)
        result_start = sum(i.start if isinstance(i, Interval) else i for i in intervals)
        result_end = sum(i.end if isinstance(i, Interval) else i for i in intervals)

        # print(result_start, result_end)
        if result_start == result_end:
            return result_start
        else:
            return Interval(result_start, result_end)


    def __interval_mul(self, *intervals):
        """
        Умножение интервалов.
        """
        intervals = [i for i in intervals if i != EmptySet]
        # print('\tFROM MUL', intervals)
        first = intervals[0]
        result_start = min(first.start, first.end) if isinstance(first, Interval) else first
        result_end = max(first.start, first.end) if isinstance(first, Interval) else first

        # print(f'\tresult_start = {result_start}, result_end = {result_end}')
        for i in range(1, len(intervals)):
            val = intervals[i]
            # print('\tval', val)
            if isinstance(val, Interval):
                result_start = min(result_start * val.start, result_start * val.end)
                result_end = max(result_end * val.start, result_end * val.end)
                # print("\tINTERVAL", result_start, result_end)
            else:
                result_start *= val
                result_end *= val
                # print("\tNUMBER", result_start, result_end)

        # print("\tMUL", result_start, result_end)
        if result_start == result_end:
            return result_start
        else:
            return Interval(result_start, result_end)


    def __real_pow(self, x, exponent):
        """
        Вычисляет действительное значение x^exponent.
        """
        ordinary_fraction = Fraction(exponent).limit_denominator()
        denominator = ordinary_fraction.denominator
        if denominator != 1 and x < 0:
            return -abs(x) ** exponent
        else:
            return x ** exponent

    def __interval_pow(self, base, exponent):
        """
        Возведение интервала в степень.
        """
        # print("\tPOW", base, exponent, type(base), type(exponent))
        if isinstance(base, Interval):
            exponent_float = float(exponent)
            ordinary_fraction = Fraction(exponent_float).limit_denominator()
            numenator = ordinary_fraction.numerator
            denominator = ordinary_fraction.denominator

            if exponent_float > 0:
                if denominator % 2 == 0:
                    if base.start >= 0:
                        # print("\tFROM POW 1", Interval(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float)))
                        return Interval(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float))
                    else:
                        # print("\tFROM POW 2", None)
                        return None
                else:
                    if numenator % 2 == 0:
                        if base.end < 0 or base.start > 0:
                            # print("\tFROM POW 3.1", Interval(min(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float)), max(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float))))
                            return Interval(min(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float)), max(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float)))
                        else:
                            # print("\tFROM POW 3.2", Interval(0, max(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float))))
                            return Interval(0, max(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float)))
                    else:
                        # print("\tFROM POW 4", Interval(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float)))
                        return Interval(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float))
            else:
                if denominator % 2 == 0:
                    if base.end <= 0:
                        # print("\tFROM POW 5", None)
                        return None
                    elif base.start == 0:
                        # print("\tFROM POW 6", Interval(self.__real_pow(base.end, exponent_float), self.inf))
                        return Interval(self.__real_pow(base.end, exponent_float), self.inf)
                    else:
                        # print("\tFROM POW 7", Interval(self.__real_pow(base.end, exponent_float), self.__real_pow(base.start, exponent_float)))
                        return Interval(self.__real_pow(base.end, exponent_float), self.__real_pow(base.start, exponent_float))
                else:
                    if numenator % 2 == 0:
                        if base.end < 0 or base.start > 0:
                            # print("\tFROM POW 8", Interval(min(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float)), max(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float))))
                            return Interval(min(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float)), max(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float)))
                        elif base.start == 0:
                            # print("\tFROM POW 9", Interval(self.__real_pow(base.end, exponent_float), self.inf))
                            return Interval(self.__real_pow(base.end, exponent_float), self.inf)
                        elif base.end == 0:
                            # print("\tFROM POW 10", Interval(self.__real_pow(base.start, exponent_float), self.inf))
                            return Interval(self.__real_pow(base.start, exponent_float), self.inf)
                        else:
                            # print("\tFROM POW 11", Interval(min(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float)), self.inf))
                            return Interval(min(self.__real_pow(base.start, exponent_float), self.__real_pow(base.end, exponent_float)), self.inf)
                    else:
                        if base.end < 0 or base.start > 0:
                            # print("\tFROM POW 12", Interval(self.__real_pow(base.end, exponent_float), self.__real_pow(base.start, exponent_float)))
                            return Interval(self.__real_pow(base.end, exponent_float), self.__real_pow(base.start, exponent_float))
                        elif base.start == 0:
                            # print("\tFROM POW 13", Interval(self.__real_pow(base.end, exponent_float), self.inf))
                            return Interval(self.__real_pow(base.end, exponent_float), self.inf)
                        elif base.end == 0:
                            # print("\tFROM POW 14", Interval(-self.inf, self.__real_pow(base.start, exponent_float)))
                            return Interval(-self.inf, self.__real_pow(base.start, exponent_float))
                        else:
                            # print("\tFROM POW 15", Interval(-self.inf, self.inf))
                            return Interval(-self.inf, self.inf)
        else:
            # print("\tFROM POW 16", float(base**exponent))
            return float(base**exponent)


    def __interval_sin(self, arg):
        """
        Вычисление синуса для интервала.
        """
        if isinstance(arg, Interval):
            res_start = np.min([sp.sin(arg.start), sp.sin(arg.end)])
            res_end = np.max([sp.sin(arg.start), sp.sin(arg.end)])
            if abs(arg.start % (2 * sp.pi) - 3 * sp.pi / 2) < 0.01:
                res_start = -1
            if abs(arg.end % (2 * sp.pi) - sp.pi / 2) < 0.01:
                res_end = 1
            if res_start == res_end:
                return res_start
            else:
                return Interval(res_start, res_end)
        else:
            return sp.sin(arg)


    def __interval_cos(self, arg):
        """
        Вычисление косинуса для интервала.
        """
        if isinstance(arg, Interval):
            res_start = np.min([sp.cos(arg.start), sp.cos(arg.end)])
            res_end = np.max([sp.cos(arg.start), sp.cos(arg.end)])
            if abs((arg.start % sp.pi) - 0) < 0.01:
                res_end = 1
            if abs((arg.end % sp.pi) - sp.pi) < 0.01:
                res_start = -1
            if res_start == res_end:
                return res_start
            else:
                return Interval(res_start, res_end)
        else:
            return sp.cos(arg)


    def __interval_exp(self, arg):
        """
        Вычисление экспоненты для интервала.
        """
        if isinstance(arg, Interval):
            return Interval(sp.exp(arg.start), sp.exp(arg.end))
        else:
            return sp.exp(arg)
