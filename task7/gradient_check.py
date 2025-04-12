import numpy as np
import torch

GRADIENT_CHECK_DELTA = 1e-3


def check_gradient(f, x, delta=GRADIENT_CHECK_DELTA, tol=1e-3):
    """
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    """
    assert isinstance(x, torch.Tensor)
    x = x.clone().numpy()

    orig_x = x.copy()
    fx, analytic_grad, is_predefined_grad = f(x)
    fx = fx.clone().numpy()
    is_predefined_grad = is_predefined_grad.clone().numpy()
    analytic_grad = analytic_grad.clone().numpy()
    assert np.all(np.isclose(orig_x, x, tol)), (
        "Functions shouldn't modify input variables"
    )

    assert analytic_grad.shape == x.shape

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        if is_predefined_grad[ix]:
            it.iternext()
            continue
        analytic_grad_at_ix = analytic_grad[ix]
        h = np.zeros(x.shape)
        h[ix] = delta

        fxh, _, _ = f(x + h)
        fxh = fxh.clone().numpy()
        numeric_grad_at_ix = (fxh[ix] - fx[ix]) / delta

        if (abs(numeric_grad_at_ix - analytic_grad_at_ix) > tol).any():
            print(
                "Gradients are different at %s. Analytic: %s, Numeric: %s"
                % (ix, analytic_grad_at_ix, numeric_grad_at_ix)
            )
            return False

        it.iternext()

    # print("Gradient check passed!", end="|")
    return True
