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

