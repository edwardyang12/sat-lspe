# sat-lspe
We present SAT-LSPE which combines [SAT](https://github.com/BorgwardtLab/SAT) and [GNN-LSPE](https://github.com/vijaydwivedi75/gnn-lspe) to increase the positional and structural awareness of our Graph Neural Network and test it on two molecular graph regression tasks: ZINC and AQSOL
## Repository Setup
To setup the datasets for experiments, we assume that there is a copy of the dataset from `torch_geometric.datasets` on your device. After downloading, please run `python data/molecules.py` to setup and preprocess the datasets to be used. We assume that the preprocessed datasets are also in the `data` directory for our experiments.

## Reproducibility
To run our method, simply call `python` on either `main_aqsol.py` or `main_zinc.py` depending on which dataset you want to evaluate on. In addition, please provide the location of the configs you want to use to the `--config` parameter. A full set of configurable options are provided in each of the "main" python files and can also be found in the `configs` directory. For example, to run SAT-LSPE on the `ZINC` dataset, we could use the following command:

```
python main_zinc.py --config 'config/sat_lspe.json' --model SATLSPE --out_dir 'out/sat-lspe/' --dataset ZINC 
```

We include sample jobs that provide further detail on how we ran the experiments in the `jobs` directory. Or, to run a full set of experiments, you can use `all.sh` which will run SAT, LSPE, and SAT-LSPE on both datasets.

## Output
Output results are located in the directory defined by the `out_dir` config parameter. In this directory, there will be a `results` directory which contains the results in a text file format, a `checkpoints` directory which contains the model checkpoints, and a `logs` directory which contains the tensorboard logs for that specific experiment.