# Time Fairness in Online Knapsack Problems

(abstract here)

# Python code 

Our experimental code has been written in Python.  We recommend using a tool to manage Python virtual environments, such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  There are several required Python packages:
- [NumPy](https://numpy.org)
- [SciPy](https://scipy.org)
- [tqdm](https://github.com/tqdm/tqdm) for CLI progress bar
- [Matplotlib](https://matplotlib.org) for creating plots 
- [seaborn](https://seaborn.pydata.org) for creating plots 
- [mat4py](https://pypi.org/project/mat4py/) for importing data from MATLAB 

# Files and Descriptions

1. **knapsack.py** ([see Section 4]()): description
2. **load_traces.py**: description
3. **experimentsFluctuation.py**: description
4. **experimentsError.py**: description
5. **data-cloud**: This folder contains MATLAB code to generate knapsack sequences from the cloud trace data set.  Existing traces will be automatically loaded into Python if running ``experiments*.py``.

## Dataset References

**Google cloud trace data set:**

Charles Reiss, Alexey Tumanov, Gregory R. Ganger, Randy H. Katz, and Michael A. Kozuch. 2012. Heterogeneity and dynamicity of clouds at scale: Google trace analysis. In Proceedings of the Third ACM Symposium on Cloud Computing (SoCC '12). Association for Computing Machinery, New York, NY, USA, Article 7, 1–13. https://doi.org/10.1145/2391229.2391236

# Reproducing Results

Given a correctly configured Python environment, with all of the described dependencies installed, one can reproduce our results by cloning this repository, and running either of the following in a command line at the root directory, for experiments and real-world networks, respectively:

- Changing value fluctuation: `` python3 experimentsFluctuation.py ``
- Changing simulated prediction error: `` python3 experimentsError.py ``


# Citation

> @misc{lechowicz2023fairknapsack, 
> title={Time Fairness in Online Knapsack Problems},
> author={Adam Lechowicz and Rik Sengupta and Bo Sun and Shahin Kamali and Mohammad Hajiesmaili},
> eprint={2305.XXXXX},
> year={2023},
> archivePrefix={arXiv},
> primaryClass={cs.SI}}
