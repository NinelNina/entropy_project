import numpy as np

import entropy
import probability


def input_array():
    rows = int(input("Введите количество X: "))
    cols = int(input("Введите количество Y: "))

    array = np.empty((rows, cols))

    for i in range(rows):
        for j in range(cols):
            element = float(input(f"Введите элемент матрицы XY[{i+1}][{j+1}]: "))
            array[i][j] = element

    print("Двумерная матрица совместных вероятностей XY:")
    print(array)

    return array


def calculate_all_characteristics(p_XY):
    p_X, p_Y, str_res = probability.calc_probability(p_XY)
    p_of_X_given_Y, p_of_Y_given_X, res = probability.calc_conditional_probability(p_XY, p_X, p_Y)
    str_res += res
    H_X, H_Y, res = entropy.calc_binary_entropy(p_X, p_Y)
    str_res += res
    H_XY, res = entropy.calc_joint_entropy(p_XY)
    str_res += res
    H_X_by_Y, H_Y_by_X, res = entropy.calc_full_conditional_entropy(p_X, p_Y, p_of_X_given_Y, p_of_Y_given_X)
    str_res += res
    return str_res


def main_func():
    p_XY = input_array()
    result = "p(XY) = \n" + str(p_XY) + '\n'
    result += calculate_all_characteristics(p_XY)

    with open("1.txt", "w") as file:
        file.write(result)


if __name__ == '__main__':
    main_func()

