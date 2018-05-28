# cracking-rnn
A recurrent neural network for password cracking.



Separate dataset into train/valid/test: 

```sh
$ python preprocess.py
```

Note that preprocess.py does **not** shuffle its input. You should shuffle your dataset before feeding it to preprocess.py



Train your own model: 

```sh
$ python train.py
```



Sample all passwords with possibility above threshold: 

```sh
$ python sample.py [--threshold <threshold>]
```



Evaluate how well your model performed: 

```sh
$ python eval.py <results_file> <test_data_file>
```



## Estimating Guess Numbers with Monte Carlo

Sampling a large number of passwords may take a **very** long time. Therefore, we use Monte Carlo methods to estimate the guess number of each password in the test set. 

Guess number: number of guesses needed to crack a particular password. 



The code for Monte Carlo requires two input files: one that contains k possibilities randomly sampled from the entire distribution (e.g. k = 10000) and another that contains the possibility of each password in the test set. 



Randomly sample k possibilities from the distribution: 

```bash
$ python sample-for-monte-carlo.py [--sample_size <sample_size>]
```



Assign possibilities to each password in the test set: 

```bash
$ python assign-probs.py
```



Evaluate how well your model performed: 

```bash
$ python eval-monte-carlo.py
```



Note that due to copyright reasons, the code for Monte Carlo is not included in this repository. 

