# Online Conversion with Switching Costs: Robust and Learning-Augmented Algorithms

[<img src="https://img.shields.io/badge/Full%20Paper-2310.20598-B31B1B.svg?style=flat-square&logo=arxiv" height="25">](https://arxiv.org/abs/2310.20598)

We introduce and study online conversion with switching costs, a family of online problems that capture emerging problems at the intersection of energy and sustainability. In this problem, an online player attempts to purchase (alternatively, sell) fractional shares of an asset during a fixed time horizon with length $T$. At each time step, a cost function (alternatively, price function) is revealed, and the player must irrevocably decide an amount of asset to convert. The player also incurs a switching cost whenever their decision changes in consecutive time steps, i.e., when they increase or decrease their purchasing amount. We introduce competitive (robust) threshold-based algorithms for both the minimization and maximization variants of this problem, and show they are optimal among deterministic online algorithms. We then propose learning-augmented algorithms that take advantage of untrusted black-box advice (such as predictions from a machine learning model) to achieve significantly better average-case performance without sacrificing worst-case competitive guarantees. Finally, we empirically evaluate our proposed algorithms using a carbon-aware EV charging case study, showing that our algorithms substantially improve on baseline methods for this problem.

# Python code

Our experimental code has been written in Python and Cython.  We recommend using a tool to manage Python virtual environments, such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  There are several required Python packages:
- [Cython](https://cython.org)
- [NumPy](https://numpy.org)
- [pandas](https://pandas.pydata.org)
- [SciPy](https://scipy.org)
- [tqdm](https://github.com/tqdm/tqdm) for progress bars
- [Matplotlib](https://matplotlib.org) for creating plots 
- [Seaborn](https://seaborn.pydata.org)

# Files and Descriptions

1. **functions2.pyx**: Implements shared functions and algorithms in Cython
2. **experiments.py**: Main Python script for experiments
3. **advice_experiments.py**: Python script for advice experiments (varying adversarial factor $\xi$)
4. **setup.py**: Setup script for Cython compilation
5. **solar_loader.py**: Implements simple logic to convert solar radiation data to estimated PV electricity generation traces
6. **supplemental.py**: Implements occasionally used shared functions
7. **ForecastsCISO.csv**: Carbon intensity data and machine-learned forecasts for CAISO grid region given by [CarbonCast](https://github.com/carbonfirst/CarbonCast)
8. **solar_radiation.csv**: Solar radiation data for the experiment location given by the [NSRDB](https://nsrdb.nrel.gov)
9. **acndata_sessions.json**: EV charging session traces from [ACN-Data](https://ev.caltech.edu/dataset)
10. **plots/**: directory, contains CDF plots from main experiment  (experiments.py)
11. **advice_plots/**: directory, contains heatmap plot from advice experiment (advice_experiments.py)

# Dataset References

**Carbon Intensity Data:**

Electricity Maps. retrieved 2023. https://www.electricitymaps.com

**Carbon Intensity Forecasts:**

Diptyaroop Maji, Prashant Shenoy, and Ramesh K. Sitaraman. 2022. CarbonCast: multi-day forecasting of grid carbon intensity. In Proceedings of the 9th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys '22). Association for Computing Machinery, New York, NY, USA, 198–207. https://doi.org/10.1145/3563357.3564079

**Historical Solar Radiation Data:**

Manajit Sengupta, Yu Xie, Anthony Lopez, Aron Habte, Galen Maclaurin, James Shelby, 2018. The National Solar Radiation Data Base (NSRDB). Renew. Sustain. Energy Rev. 89, 51-60. https://doi.org/10.1016/j.rser.2018.03.003.

**EV Charging Data set:**

Zachary J. Lee, Tongxin Li, and Steven H. Low. 2019. ACN-Data: Analysis and Applications of an Open EV Charging Dataset. In Proceedings of the Tenth ACM International Conference on Future Energy Systems (e-Energy '19). Association for Computing Machinery, New York, NY, USA, 139–149. https://doi.org/10.1145/3307772.3328313

# Reproducing Results (FINISH)

Given a correctly configured Python environment, with all dependencies, one can reproduce our results by cloning this repository, and running either of the following in a command line at the root directory, for synthetic and real-world networks, respectively:

``python experiments.py {TRACE CODE} {AMOUNT OF SLACK}``

Pass the abbreviation for the trace file and the desired amount of slack as command line arguments.  For example, running the experiments on the Ontario trace with 48 hours of slack is accomplished by running ``python experiments.py CA 48``

# Citation

> @inproceedings{lechowicz2024online,
> title        = {{Online Conversion with Switching Costs: Robust and Learning-augmented Algorithms}},
> author       = {Adam Lechowicz and Nicolas Christianson and Bo Sun and Noman Bashir and Mohammad Hajiesmaili and Adam Wierman and Prashant Shenoy},
> year         = 2024,
> month        = {Jun},
> booktitle    = {Proceedings of the 2024 SIGMETRICS/Performance Joint International Conference on Measurement and Modeling of Computer Systems},
> location     = {Venice, Italy},
> publisher    = {Association for Computing Machinery},
> address      = {New York, NY, USA},
> series       = {SIGMETRICS / Performance '24} }
