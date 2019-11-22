# TimeAwareRNN

This repository provides the data and code for running the experiments described in 
the paper [System Identification with Time-Aware Neural Sequence Models](https://arxiv.org/pdf/1911.09431.pdf) (published at AAAI 2020).
The software has been tested with python 3.7.3 and pytorch 1.1.0.

For now, I've uploaded (a slightly thinned out version of) the *research code* (you know what I mean).
It still has command line options you'd never use (e.g., for the artificial baselines).
However, I think some of the code can be used for further experimentation on modeling time series with recurrent neural networks.

In particular, the `taho` package (for*time-aware higher-order*) contains code for 
- batch training of recurrent neural networks for generic 
Multiple Input - Multiple Output (MIMO) system data under the form of a single time series 
(similar to scheme 4 from [Character-level Recurrent Neural Networks in Practice: Comparing Training and
Sampling Schemes](https://arxiv.org/pdf/1801.00632.pdf))
- a basic MIMO neural network model for ad hoc recurrent cells 
- the time-aware (and higher-order) extension of the GRU (and a simple implementation of the anti-symmetric RNN)

Feel free to use what you need from the code. If you do, please cite the following paper:

*Demeester, T. 2020. "System Identification with Time-Aware Neural Sequence Models", 
In 34th AAAI Conference on Artificial Intelligence (AAAI 2020), New York, USA. AAAI Press.* 

Don't hesitate to drop me an email (see the header of the paper) if you have questions of suggestions. 



### Experiments

I used the Continuous Stirred Tank Reactor (CSTR) data, 
as well as the data from a Test Setup of an Industrial Winding Process (Winding), both from the [DaISy](https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html) dataset.
The `main.py` files in both the `CSTR` and `winding` folder are largely similar, but I kept them separate for now.
For convenience I've already included the data files with the normalized values and missing data 
(in the respective `data` subfolders within `CSTR` and `winding`).  

The results for Table 4 (time-aware higher-order GRU extension) can be obtained as follows.

For CSTR, from within the CSTR folder:
```console
# RK4 scheme
python main.py --model GRU --time_aware variable --scheme RK4 --k_state 20 --missing 0.50 --lr 0.001 --batch_size 512
```

For winding, from within the winding folder:
```console
# RK4 scheme, linear input interpolation
python main.py --model GRU --time_aware variable --scheme RK4 --interpol linear --k_state 10 --missing 0.50 --lr 0.003 --batch_size 512
```

Note that the reported results are the average over 5 runs with different random seeds.
Given the provided standard deviation, the obtained values may therefore deviate from the reported ones.

The following options are needed for the baselines:
-  `--time_aware` set to `no` for time-unaware models, or `input` for using the interval width as input feature in a fixed-width scheme.
-  `--model` set to `GRU`, `GRUinc`, `ARNN`, or `ARNNinc`, with `ARNN` for the anti-symmetric RNN, and `inc` for the incremental (non-stationary) schemes.

