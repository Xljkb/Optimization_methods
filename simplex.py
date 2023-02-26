import heapq
import numpy as np


n:  int = 2          # size
m:  float = 0.5      # Simplex edge length
x1: int = 1          # first point
x2: int = 1          # second point

epsilon: float = 0.1  # accuracy

delta1 = ((np.sqrt(n+1) - 1)/(n * np.sqrt(2))) * m
delta2 = ((np.sqrt(n+1) + n - 1)/(n * np.sqrt(2))) * m


def func(x, y):
    """
    Function to optimize.
    :param x: first point of function.
    :param y: second point of function.
    :return: value of function.
    """
    return 2.8 * np.square(y) + 1.9 * x + 2.7 * np.square(x) + 1.6 - 1.9 * y


def primary_matrix():
    """
    Primary matrix for first 3 values of function.
    I's calculating especially for first rows, that used deltas.
    :return: main_array (Primary matrix)
    """

    row1 = [x1, x2]
    row1.append(func(row1[0], row1[1]))

    row2 = [x1 + delta1, x2 + delta2]
    row2.append(func(row2[0], row2[1]))

    row3 = [x1 + delta2, x2 + delta1]
    row3.append(func(row3[0], row3[1]))

    main_array = np.array([row1, row2, row3])

    return main_array


def calculate():
    """
    Calculating of Simplex/.
    :return: main_array (Matrix with all steps)
    """
    main_array = primary_matrix()
    count = 0
    check = True  # termination condition

    while check:

        def sort_key(e):
            """
            Sorting method that sort with value of function.
            :param e: matrix row.
            :return: function value of row F(xi).
            """
            return e[n]

        lowest_row, middle_low_row, highest_low_row = heapq.nsmallest(3, main_array, key=sort_key)

        xc_1 = (1 / (n + 1)) * (lowest_row[0] + middle_low_row[0] + highest_low_row[0])
        xc_2 = (1 / (n + 1)) * (lowest_row[1] + middle_low_row[1] + highest_low_row[1])
        xcwhole = [xc_1, xc_2]  # Центр тяжести симплекса

        xc1 = (1 / n) * (lowest_row[0] + middle_low_row[0])
        xc2 = (1 / n) * (middle_low_row[1] + lowest_row[1])
        xc = [xc1, xc2]
        f_xc = func(xcwhole[0], xcwhole[1])

        current_row = [(2 * xc[0] - highest_low_row[0]), 2 * xc[1] - highest_low_row[1]]
        current_row.append(func(current_row[0], current_row[1]))

        if (abs(lowest_row[n] - f_xc) < epsilon) and (abs(middle_low_row[n] - f_xc) < epsilon) and (abs(highest_low_row[n] - f_xc) < epsilon):
            check = False

        elif current_row[n] < highest_low_row[n]:
            main_array = np.vstack([main_array, current_row])
            # print(main_array)
        elif current_row[n] >= highest_low_row[n]:
            reduction_row = [lowest_row[0] + 0.5 * (highest_low_row[0] - lowest_row[0]), lowest_row[1] + 0.5 * (highest_low_row[1] - lowest_row[1])]
            reduction_row.append(func(reduction_row[0], reduction_row[1]))
            main_array = np.vstack([main_array, reduction_row])

            reduction_row2 = [lowest_row[0] + 0.5 * (middle_low_row[0] - lowest_row[0]), lowest_row[1] + 0.5 * (middle_low_row[1] - lowest_row[1])]
            reduction_row2.append(func(reduction_row2[0], reduction_row2[1]))
            main_array = np.vstack([main_array, reduction_row2])
        count += 1
    print(f"Total iterations: {count}")

    return main_array


print(calculate())
