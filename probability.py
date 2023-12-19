import numpy as np


global round_to
round_to = 3


def is_independent(p_XY, p_X, p_Y):
    n, m = p_XY.shape
    str_res = ''
    tmp_res = ''
    is_independent = True
    for i in range(n):
        for j in range(m):
            tmp = round(p_X[i] * p_Y[j], round_to)
            str_res += ('p(x{}) * p(y{}) = '.format(i + 1, j + 1) + str(p_X[i]) + ' * ' + str(p_Y[j]) + ' = '
                        + str(tmp) + '\n')

            if tmp != p_XY[i][j]:
                tmp_res += ('p(x{}) * p(y{}) = '.format(i + 1, j + 1) + str(tmp) + ' != '
                           + str(p_XY[i][j]) + ' != p(x{}y{})'.format(i + 1, j + 1) + '\n')
                is_independent = False
            else:
                tmp_res += ('p(x{}) * p(y{}) = '.format(i + 1, j + 1) + str(p_X[i]) + ' * ' + str(p_Y[j]) + ' = '
                        + str(p_XY[i][j]) + ' = p(x{}y{})'.format(i + 1, j + 1)) + '\n'
    if is_independent:
        tmp_res += 'Ансамбли независимы.\n'
    else:
        tmp_res += 'Ансамбли зависимы.\n'
    str_res += tmp_res
    return str_res


def calc_probability(p_XY):
    n, m = p_XY.shape

    str_res = ''
    str_num = ''

    p_X = np.zeros(n)
    p_Y = np.zeros(m)

    for i in range(n):
        str_res += 'p(x{}) = '.format(i + 1)
        for j in range(m):
            str_res += 'p(x{}y{})'.format(i + 1, j + 1)
            str_num += str(p_XY[i][j])
            p_X[i] += p_XY[i][j]

            if j < m - 1:
                str_res += ' + '
                str_num += ' + '
            else:
                str_res += ' = '
                p_X[i] = round(p_X[i], round_to)
                str_res += str_num + ' = ' + str(p_X[i]) + '\n'
                str_num = ''

    for j in range(m):
        str_res += 'p(y{}) = '.format(j + 1)
        for i in range(n):
            str_res += 'p(x{}y{})'.format(i + 1, j + 1)
            str_num += str(p_XY[i][j])
            p_Y[j] += p_XY[i][j]
            if i < n - 1:
                str_res += ' + '
                str_num += ' + '
            else:
                p_Y[j] = round(p_Y[j], round_to)
                str_res += ' = '
                str_res += str_num + ' = ' + str(p_Y[j]) + '\n'
                str_num = ''

    str_res += is_independent(p_XY, p_X, p_Y)

    return p_X, p_Y, str_res


def calc_conditional_probability(p_XY, p_X, p_Y):
    n, m = p_XY.shape

    p_of_X_given_Y = np.empty((n, m))
    p_of_Y_given_X = np.empty((m, n))

    str_res = ''

    for i in range(n):
        for j in range(m):
            str_res += 'p(x{}|y{}) = '.format(i + 1, j + 1) + 'p(x{}y{})'.format(i + 1, j + 1) + '/' + 'p(y{})'.format(
                j + 1) + ' = '
            str_res += str(p_XY[i][j]) + '/' + str(p_Y[j]) + ' = '
            p_of_X_given_Y[i][j] = round(p_XY[i][j] / p_Y[j], round_to)
            str_res += str(p_of_X_given_Y[i][j]) + '\n'

    for j in range(m):
        for i in range(n):
            str_res += 'p(y{}|x{}) = '.format(j + 1, i + 1) + 'p(x{}y{})'.format(i + 1, j + 1) + '/' + 'p(x{})'.format(
                i + 1) + ' = '
            str_res += str(p_XY[i][j]) + '/' + str(round(p_X[i], round_to)) + ' = '
            p_of_Y_given_X[j][i] = round(p_XY[i][j] / p_X[i], round_to)
            str_res += str(p_of_Y_given_X[j][i]) + '\n'

    return p_of_X_given_Y, p_of_Y_given_X, str_res

