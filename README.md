# Time Fairness in Online Knapsack Problems

The online knapsack problem is a classic problem in the field of online algorithms. Its canonical version asks how to pack items of different values and weights arriving online into a capacity-limited knapsack so as to maximize the total value of the admitted items. Although optimal competitive algorithms are known for this problem, they may be fundamentally unfair, i.e., individual items may be treated inequitably in different ways. Inspired by recent attention to fairness in online settings, we develop a natural and practically-relevant notion of time fairness for the online knapsack problem, and show that the existing optimal algorithms perform poorly under this metric. We propose a parameterized deterministic algorithm where the parameter precisely captures the Pareto-optimal trade-off between fairness and competitiveness. We show that randomization is theoretically powerful enough to be simultaneously competitive and fair; however, it does not work well in practice, using trace-driven experiments. To further improve the trade-off between fairness and competitiveness, we develop a fair, robust (competitive), and consistent learning-augmented algorithm with substantial performance improvement in trace-driven experiments.

# Python code 

Our experimental code has been written in Python.  We recommend using a tool to manage Python virtual environments, such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  There are several required Python packages:
- [NumPy](https://numpy.org)
- [SciPy](https://scipy.org)
- [tqdm](https://github.com/tqdm/tqdm) for CLI progress bar
- [Matplotlib](https://matplotlib.org) for creating plots 
- [seaborn](https://seaborn.pydata.org) for creating plots 
- [mat4py](https://pypi.org/project/mat4py/) for importing data from MATLAB 

# Files and Descriptions

1. **knapsack.py** (see Section 4): contains Python implementations of each tested knapsack algorithm, including a dynamic programming optimal solution (note that the DP solution requires integer weights), alongside $\mathsf{ZCL}$, $\mathsf{ECT}$, and LA-ECT.
2. **load_traces.py**: loads traces from ``.mat`` files located in ``data-cloud``, computes optimal solutions and $d^*$ values for each trace, and saves traces to a serialized file on disk.
3. **experiments.py**: code for first experiment (see Section 5), which tests algorithms not using predictions with several values of U/L, then plots CDFs of the empirical competitive ratios.
4. **experimentsLA.py**: code for second experiment (see Section 5), which tests learning-augmented algorithms with several different error values in prediction, then plots CDFs of the empirical competitive ratios.
5. **data-cloud**: This folder contains MATLAB code to generate knapsack sequences from the cloud trace data set.  Existing traces will be automatically loaded into Python if running ``experiments*.py``.

## Dataset References

**Google cloud trace data set:**

Charles Reiss, Alexey Tumanov, Gregory R. Ganger, Randy H. Katz, and Michael A. Kozuch. 2012. Heterogeneity and dynamicity of clouds at scale: Google trace analysis. In Proceedings of the Third ACM Symposium on Cloud Computing (SoCC '12). Association for Computing Machinery, New York, NY, USA, Article 7, 1–13. https://doi.org/10.1145/2391229.2391236

# Reproducing Results

Given a correctly configured Python environment, with all of the described dependencies installed, one can reproduce our results by cloning this repository, and running either of the following in a command line at the root directory, for standard experiments and learning-augmented experiments, respectively:

- Changing value density bounds $U/L$: `` python3 experiments.py ``
- Changing simulated prediction error: `` python3 experimentsLA.py ``


# Citation

> @misc{lechowicz2023fairknapsack, 
> title={Time Fairness in Online Knapsack Problems},
> author={Adam Lechowicz and Rik Sengupta and Bo Sun and Shahin Kamali and Mohammad Hajiesmaili},
> eprint={2305.XXXXX},
> year={2023},
> archivePrefix={arXiv},
> primaryClass={cs.SI}}
