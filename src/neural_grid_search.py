############   NATIVE IMPORTS  ###########################
from typing import List, Iterable
############ INSTALLED IMPORTS ###########################
############   LOCAL IMPORTS   ###########################
##########################################################

class NeuralGridSearch:
    @staticmethod
    def greedy_search(vector_size:int) -> Iterable[List[int]]:
        """ step through every possible combination of weights in a vector/layer (each weight is 0 or 1) """
        return map(
            lambda i:NeuralGridSearch._convert_int_to_binary_vector(number=i,vector_size=vector_size),
            range(NeuralGridSearch.max_range(vector_size=vector_size))
        )

    @staticmethod
    def float_range(min_value:int=-1, max_value:int=1,step_size:float=.01) -> float:
        """ step through a range of values with a floating point step size """
        for weight in range(
            int(min_value//step_size),
            int(max_value//step_size) + 1,
            1
        ):
            yield weight*step_size

    @staticmethod
    def max_range(vector_size:int) -> int:
        """ get the total number of combinations possible for a vector of size N weights """
        return NeuralGridSearch._convert_binary_string_to_int(
            binary_string='1'*vector_size
        ) + 1

    @staticmethod
    def _convert_binary_string_to_int(binary_string:str) -> int:
        return int(binary_string, 2)

    @staticmethod
    def _convert_int_to_binary_string(number:int,leading_zeros:int) -> str:
        return format(number,f"#0{leading_zeros+2}b")[2:]

    @staticmethod
    def _convert_int_to_binary_vector(number:int, vector_size:int) -> List[int]:
        return list(
            map(
                int,
                NeuralGridSearch._convert_int_to_binary_string(
                    number=number,
                    leading_zeros=vector_size
                )
            )
        )  