import re
from typing import List
from sympy import lambdify, sympify, latex, Interval, diff, hessian
from sympy.sets import FiniteSet
from latex2sympy2 import latex2sympy


from backend.IntervalNaturalExtention import IntervalNaturalExtention

class Function:
    """
    Класс для парсинга функции из LaTeX-строки.
    Пример использования:
    func = Function(r"\sin(x_1) + x_2^2")
    result = func([1.0, 2.0])  # 0.84147 + 4 = 4.84147
    """
    def __init__(self, str_func: str, is_grad=False, is_hess=False):
        self.is_grad = is_grad
        self.is_hess = is_hess
        self.to_beautify = ""
        self.gradient = None
        self.hessian = None
        self.expr, self.variables = self.__parse_func(str_func)
        self.interval_natural_extension = IntervalNaturalExtention()
        # self.compiled_func = self.__compile_func(self.expr, self.variables)

    def __call__(self, x: list):
        return self.__calculate_func(x)
    
    # def get_gradient(self):
    #     return self.gradient
    def get_gradient(self):
        if self.gradient is None:
            self.gradient = [diff(self.expr, var) for var in self.variables]
        return self.gradient
    
    def get_hessian(self):
        if self.hessian is None:
            self.hessian = hessian(self.expr, self.variables).tolist()
        return self.hessian
    
    def count_vars(self):
        return len(self.variables)
    
    def get_vars(self):
        return list(map(lambda x: re.sub(r"[{}_]", "", str(x)), self.variables))

    def get_latex_func(self):
        return latex(self.to_beautify)

    def __parse_func(self, str_func: str):
        # print('PARSE')
        if self.is_grad or self.is_hess:
            # print('IS_GRAD')
            expr = sympify(str_func)
            variables = sorted(expr.free_symbols, key=lambda s: s.name)
        else:
            # print('NOT_IS_GRAD')
            without_spaces = str_func.replace(" ", "")
            self.to_beautify = latex2sympy(without_spaces)
            expanded = self.__expand_sums(without_spaces)
            simplified = self.__simplify_indexes(expanded)
            expr = latex2sympy(simplified)
            variables = sorted(expr.free_symbols, key=lambda s: s.name)
            # self.gradient = [diff(expr, var) for var in variables]
            # self.hessian = hessian(expr, variables).tolist()
        # print(self.gradient)
        return expr, variables

    # def __compile_func(self, expr, variables):
    #     return lambdify(variables, expr, modules="numpy")

    def __calculate_func(self, x: list):
        # Создаем словарь для подстановки переменных
        interval_dict = {var: val for var, val in zip(self.variables, x)}
        for key, val in interval_dict.items():
            if isinstance(val, FiniteSet):
                interval_dict[key] = val.args[0]
        # Если хотя бы одна переменная — интервал, используем рекурсивную обработку
        if any(isinstance(val, Interval) for val in x):
            return self.interval_natural_extension.eval_interval_expr(self.expr, interval_dict)
            # return eval_interval_expr(self.expr, interval_dict)
        else:
            # Если все переменные — числа, используем lambdify
            compiled_func = lambdify(self.variables, self.expr, modules="numpy")
            return compiled_func(*x)
        # return float(self.compiled_func(*x))

    def __simplify_indexes(self, subexpr):
        var_pattern = re.compile(r"[xy]_{")
        matches = var_pattern.finditer(subexpr)
        simplified_subexpr = subexpr
        for match_ in list(matches)[::-1]:
            start_index = match_.start()
            end_index = match_.end()
            exp_start = end_index
            exp_end = exp_start
            count_mustaches = 1
            excess_open_index, excess_close_index = 0, 0
            while exp_end < len(subexpr) and count_mustaches != 0:
                if subexpr[exp_end] == "{":
                    excess_open_index = exp_end
                    count_mustaches += 1
                elif subexpr[exp_end] == "}":
                    if excess_open_index > 0 and subexpr[excess_open_index+1 : exp_end].isdigit():
                        excess_close_index = exp_end
                    count_mustaches -= 1
                exp_end += 1
            exp_end -= 1 
            if excess_open_index > 0 and excess_close_index > 0:
                expression = subexpr[exp_start:excess_open_index] + subexpr[excess_open_index+1 : excess_close_index] + subexpr[excess_close_index+1:exp_end]
            else:
                expression = subexpr[exp_start : exp_end]
            result = sympify(expression)
            simplified_subexpr = simplified_subexpr[:exp_start] + str(result) + simplified_subexpr[exp_end:]
        return simplified_subexpr

    def __expand_sums(self, expr):
        sum_pattern = re.compile(r"\\sum_{")
        expanded_expr = expr
        matches = sum_pattern.finditer(expr)
        for match_ in list(matches)[::-1]:
            start_index = match_.start()
            end_index = match_.end()    # индекс бегущей переменной
            # Вычленяем границы суммирования и индексы выражения в сумме
            var = ""
            start_value = ""
            end_value = ""
            exp_start = -1
            exp_end = -1
            i = end_index
            cnt_close_mustaches = 0
            while cnt_close_mustaches < 2:
                if expr[i] == "}":
                    cnt_close_mustaches += 1
                elif cnt_close_mustaches == 0 and expr[i] == "=":
                    var = expr[end_index : i]
                elif cnt_close_mustaches == 0 and len(var) > 0:
                    start_value += expr[i]
                elif cnt_close_mustaches == 1 and expr[i] not in "{^":
                    end_value += expr[i]
                i += 1
            start_value = int(start_value)
            end_value = int(end_value)
            # Выделяем выражение в сумме
            exp_start = i
            cnt_mustaches = 0
            if expr[i] == "(":
                cnt_brackets = 1
                i += 1
                while cnt_brackets != 0:
                    if expr[i] == "(":
                        cnt_brackets += 1
                    elif expr[i] == ")":
                        cnt_brackets -= 1
                    i += 1
                if i < len(expr) and expr[i] in "^":
                    while i < len(expr):
                        if expr[i] in '+-)' and cnt_mustaches == 0:
                            break
                        if expr[i] == '{':
                            cnt_mustaches += 1
                        if expr[i] == '}':
                            cnt_mustaches -= 1
                        i += 1
            else:
                if expr[i] == "-":   # Если в начале выражения под суммой -, его допускаем
                    i += 1
                while i < len(expr):
                    if expr[i] in '+-)' and cnt_mustaches == 0:
                        break
                    if expr[i] == '{':
                        cnt_mustaches += 1
                    if expr[i] == "}":
                        cnt_mustaches -= 1
                    i += 1
            end_index = i
            # Обрабатываем выделенную сумму
            monoms = []
            for j in range(start_value, end_value + 1):
                replacement = str(j) if j < 10 else "{" + str(j) + "}"
                monoms.append(
                    expr[exp_start : end_index].replace(var, replacement)
                )
            expanded_sum = "+".join(monoms)
            result = expanded_sum
            expanded_expr = expanded_expr[:start_index] + f"({result})" + expanded_expr[end_index:]
        return expanded_expr
