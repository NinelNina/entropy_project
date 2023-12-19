import math

import probability

global round_to
round_to = probability.round_to


def binary_entropy(var_name, p_Z):
    n = p_Z.shape[0]
    str_res = 'H(' + str(var_name).upper() + ') = -('
    str_num = ''
    H_Z = 0
    for i in range(n):
        str_res += 'p(' + var_name + '{}) * '.format(i + 1) + 'log(p(' + var_name + '{}))'.format(i + 1)
        tmp = round(math.log2(p_Z[i]), round_to)
        H_Z += p_Z[i] * tmp
        str_num += str(p_Z[i]) + ' * (' + str(tmp) + ')'
        if i < n - 1:
            str_res += ' + '
            str_num += ' + '
    H_Z *= (-1)
    H_Z = round(H_Z, round_to)
    str_res += ') = -(' + str_num + ') = ' + str(H_Z) + '\n'
    return H_Z, str_res


def calc_binary_entropy(p_X, p_Y):
    H_X, str_res = binary_entropy('x', p_X)
    H_Y, res = binary_entropy('y', p_Y)
    str_res += res

    return H_X, H_Y, str_res


def calc_joint_entropy(p_XY):
    n, m = p_XY.shape
    str_res = 'H(XY) = -('
    str_num = ''
    H_XY = 0
    for i in range(n):
        for j in range(m):
            str_res += 'p(x{}y{}) * '.format(i + 1, j + 1) + 'log(p(x{}y{}))'.format(i + 1, j + 1)
            tmp = round(math.log2(p_XY[i][j]), round_to)
            H_XY += p_XY[i][j] * tmp
            str_num += str(p_XY[i][j]) + ' * (' + str(tmp) + ')'
            if j < m - 1:
                str_res += ' + '
                str_num += ' + '
        if i < n - 1:
            str_res += ' + '
            str_num += ' + '
    H_XY *= (-1)
    H_XY = round(H_XY, round_to)
    str_res += ') = -(' + str_num + ') = ' + str(H_XY) + '\n'
    return H_XY, str_res


def calc_full_conditional_entropy(p_X, p_Y, p_of_X_given_Y, p_of_Y_given_X):
    H_X_by_Y, str_res = full_conditional_entropy('x', 'y', p_Y, p_of_X_given_Y)
    H_Y_by_X, res = full_conditional_entropy('y', 'x', p_X, p_of_Y_given_X)
    str_res += res
    return H_X_by_Y, H_Y_by_X, str_res


def full_conditional_entropy(var_name1, var_name2, p_V, p_of_Z_given_V):
    n = p_V.shape[0]

    H_Z_by_V = 0
    str_res = 'H_' + var_name2 + '(' + str(var_name1).upper() + ') = '
    str_num = ''
    tmp_res = ''
    for j in range(n):
        H_Z_given_V, res = joint_conditional_entropy(var_name1, var_name2, p_of_Z_given_V, j)
        tmp_res += res
        str_res += 'p(' + var_name2 + '{})'.format(j + 1) + ' * H_' + var_name2 + '{}('.format(j + 1) + str(var_name1).upper() + ')'
        H_Z_by_V += p_V[j] * H_Z_given_V
        str_num += str(p_V[j]) + ' * ' + str(H_Z_given_V)
        if j < n - 1:
            str_res += ' + '
            str_num += ' + '

    str_res += ' = '
    H_Z_by_V = round(H_Z_by_V, round_to)
    str_res += str_num + ' = ' + str(H_Z_by_V) + '\n'

    tmp_res += str_res
    return H_Z_by_V, tmp_res


def joint_conditional_entropy(var_name1, var_name2, p_of_Z_given_V, j):
    n, m = p_of_Z_given_V.shape

    H_Z_given_V = 0
    str_res = 'H_' + var_name2 + '{}('.format(j + 1) + str(var_name1).upper() + ') = -('
    str_num = ''

    for i in range(n):
        str_res += 'p(' + var_name1 + '{}|'.format(i + 1) + var_name2 + '{}) * '.format(j + 1) + 'log(p(' + var_name1 + '{}|'.format(i + 1) + var_name2 + '{}))'.format(j + 1)
        tmp = round(math.log2(p_of_Z_given_V[i][j]), round_to)
        H_Z_given_V += p_of_Z_given_V[i][j] * tmp
        str_num += str(p_of_Z_given_V[i][j]) + ' * (' + str(tmp) + ')'
        if i < n - 1:
            str_res += ' + '
            str_num += ' + '
    H_Z_given_V *= (-1)
    H_Z_given_V = round(H_Z_given_V, round_to)
    str_res += ') = -(' + str_num + ') = ' + str(H_Z_given_V) + '\n'

    return H_Z_given_V, str_res





