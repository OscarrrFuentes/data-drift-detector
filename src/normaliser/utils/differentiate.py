#!usr/bin/env python3
"""
Different methods to differentiate
"""

import numpy as np

def calc_slope(x1: float, y1: float, x2: float, y2: float) -> tuple[float, bool]:
    """
    Calculate the slope between (x1, y1) and (x2, y2) in the x-y plane, for x1 < x2.

    Params:
        float x1: The low value of x
        float y1: The value of y and x1
        float x2: The high value of x
        float y2: The value of y at x2
    
    return:
        float: The slope calculated between the points
        bool: A flag for whether instability was detected
    """

    # Check for instability between points
    tol = 1e-8
    if abs(xdiff := x2 - x1) >= tol:
        return (y2 - y1)/xdiff, False
    else:
        if (slope := (y2 - y1)/xdiff) < 1e4:
            return slope, False
        else:
            print(f"\n\n\nWARNING: Instability detected at points {x1:.3f} and "\
                  f"{x2:.3f}, terminating minimisation procedure\n\n\n")
            return (y2 - y1)/tol, True

def central_difference(new_value: np.array, prev_values: np.array) -> tuple[np.array, float, bool]:
    """
    Binary search to find the location of a point in an array, insert it, and calculate
    the slope.

    Params:
        tuple new_value: The (x,y) value to estimate the slope at.
        np.array prev_values: The discrete distribution to find the slope for the point.
            This is a 2D array of (x,y).

    return:
        np.array updated_values: The new discrete distribution including new_value.
        float slope: The calculated slope at the point new_value. Returns None if there
            aren't enough data points.
        bool: A flag for whether instability was detected
    """

    length = len(prev_values)

    # Initial cases
    if length == 0:
        return np.array([new_value]), None, False
    if length == 1:
        if new_value[0] < prev_values[0,0]:
            return np.vstack((prev_values, new_value)), *calc_slope(*new_value, *prev_values[0])
        else:
            return np.vstack((new_value, prev_values)), *calc_slope(*prev_values[0], *new_value)
    if length == 2:
        if new_value[0] < prev_values[0,0]:
            return np.vstack((new_value, prev_values)), *calc_slope(*new_value, *prev_values[1])
        if new_value[0] < prev_values[1,0]:
            return np.insert(prev_values, 1, new_value, 0), *calc_slope(*prev_values[0], *prev_values[1])
        else:
            return np.vstack((prev_values, new_value)), *calc_slope(*prev_values[1], *new_value)

    # Binary search for the index of the x component of new_value in prev_values
    print(f"Length: {length}")
    i = 0
    idx = (length - 1) // 2
    while i <= length:
        # Edge cases, either forwards or backwards difference depending on which side
        if idx == length - 1:
            return np.vstack((prev_values, new_value)), *calc_slope(*prev_values[-1], *new_value)
        if idx == 0:
            return np.vstack((new_value, prev_values)), *calc_slope(*new_value, *prev_values[0])

        x_lo = prev_values[idx, 0]
        x_hi = prev_values[idx + 1, 0]
        # Test for multiple entries at the same x-value
        if (new_value[0] == x_lo) or (new_value[0] == x_hi):
            print("WARNING: Can't have multi-valued function")
            break
    
        if (x_lo < new_value[0]) and (new_value[0] < x_hi):
            break
        if new_value[0] < x_lo:
            idx -= max(1, length // (2**(i+2)))
        if new_value[0] > x_hi:
            idx += max(1, length // (2**(i+2)))
        i += 1
    print(idx)
    return np.insert(prev_values, idx + 1, new_value, 0), *calc_slope(*prev_values[idx], *prev_values[idx + 1])
