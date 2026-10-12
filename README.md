# Pricing Reusable Resources

Code for the paper “The Power of Static Pricing for Reusable Resources” by Adam N. Elmachtoub and Jiaqi Shi.

Paper: https://arxiv.org/abs/2302.11723

## Files

- `linear + exponential.py`: numerical calculations for Table 2.
- `Table3-4.ipynb`: reproduces the examples and upper bounds in Tables 3 and 4.
- `reusable_pricing.py`: numerical routines used by `Table3-4.ipynb`.
- `MHR-0.9755.ipynb`: certified MHR lower bounds in Table 5 and Figure 2.

## Requirements

Python 3 with NumPy, SciPy, CVXPY, mpmath, pandas, and Matplotlib.

Run each notebook from top to bottom. Keep `Table3-4.ipynb` and
`reusable_pricing.py` in the same directory.
