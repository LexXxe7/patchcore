from bisect import bisect

import numpy as np


def trapezoid(x, y, x_max=None):
    """
    This function calculates the definite integral of a curve given by
    x- and corresponding y-values. In contrast to, e.g., "np.trapz()", this
    function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x- or y-value will be ignored with a
    warning.

    Args:
        x: Samples from the domain of the function to integrate.
           Need to be sorted in ascending order. May contain the same value
           multiple times. In that case, the order of the corresponding
           y-values will affect the integration with the trapezoidal rule.
        y: Values of the function corresponding to x-values.
        x_max: Upper limit of the integration. The y-value at x_max will be
               determined by interpolation between its neighbours. Must not lie
               outside of the range of x.

    Returns:
        Area under the curve.
    """
    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print(
            """WARNING: Not all x- and y-values passed to trapezoid() are finite.
                 Will continue with only the finite values."""
        )
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if x_max is not an element of x.
    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion point cannot be zero or len(x).
            assert 0 < ins < len(x), "x_max must be between the minimum and the maximum"

            # Calculate the correction term which is the integral between the last
            # x[ins - 1] and x_max. Since we do not know the exact value of y at x_max,
            # we interpolate between y[ins] and y[ins - 1].
            y_interpolated = y[ins - 1] + (
                (y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
            )
            correction = 0.5 * (y_interpolated + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction
