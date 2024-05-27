# Online Conversion with Switching Costs: Robust and Learning-Augmented Algorithms

[<img src="https://img.shields.io/badge/Full%20Paper-2310.20598-B31B1B.svg?style=flat-square&logo=arxiv" height="25">](https://arxiv.org/abs/2310.20598)

We introduce and study online conversion with switching costs, a family of online problems that capture emerging problems at the intersection of energy and sustainability. In this problem, an online player attempts to purchase (alternatively, sell) fractional shares of an asset during a fixed time horizon with length $T$. At each time step, a cost function (alternatively, price function) is revealed, and the player must irrevocably decide an amount of asset to convert. The player also incurs a switching cost whenever their decision changes in consecutive time steps, i.e., when they increase or decrease their purchasing amount. We introduce competitive (robust) threshold-based algorithms for both the minimization and maximization variants of this problem, and show they are optimal among deterministic online algorithms. We then propose learning-augmented algorithms that take advantage of untrusted black-box advice (such as predictions from a machine learning model) to achieve significantly better average-case performance without sacrificing worst-case competitive guarantees. Finally, we empirically evaluate our proposed algorithms using a carbon-aware EV charging case study, showing that our algorithms substantially improve on baseline methods for this problem.

# Python code (FINISH)

Our experimental code has been written in Python and Cython.  We recommend using a tool to manage Python virtual environments, such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  There are several required Python packages:
- [Cython](https://cython.org)
- [NumPy](https://numpy.org)
- [pandas](https://pandas.pydata.org)
- [SciPy](https://scipy.org)
- [Matplotlib](https://matplotlib.org) for creating plots 
- [Seaborn](https://seaborn.pydata.org)

# Files and Descriptions (FINISH)

1. **functions.py**: Implements helper functions and algorithms.
2. **experiments.py**: Main Python script for all experiments.
3. **alpha\*.pickle**: Caches a dictionary of precomputed alpha values for k-min search algorithm and online pause and resume algorithm
4. **carbon-traces/**: directory, contains carbon traces in .csv format.
    - "CA": "CA-ON.csv" (Ontario, Canada)
    - "NZ": "NZ-NZN.csv" (New Zealand)
    - "US": "US-NW-PACW.csv" (Pacific NW, USA)

# Reproducing Results (FINISH)

Given a correctly configured Python environment, with all dependencies, one can reproduce our results by cloning this repository, and running either of the following in a command line at the root directory, for synthetic and real-world networks, respectively:

``python experiments.py {TRACE CODE} {AMOUNT OF SLACK}``

Pass the abbreviation for the trace file and the desired amount of slack as command line arguments.  For example, running the experiments on the Ontario trace with 48 hours of slack is accomplished by running ``python experiments.py CA 48``
