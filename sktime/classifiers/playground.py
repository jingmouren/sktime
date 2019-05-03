from joblib import Parallel, delayed

from classifiers.proximity import ProximityForest
from contrib import experiments
from contrib.experiments import run_experiment
from datasets import load_gunpoint
from datasets.base import load_italy_power_demand, load_arrow_head
from utils import utilities
import numpy as np

def run_pf(seed, dataset_loader):
    x_train, y_train = dataset_loader(split='TRAIN', return_X_y=True)
    x_test, y_test = dataset_loader(split='TEST', return_X_y=True)
    classifier = ProximityForest(rand=np.random.RandomState(seed), r=5, num_trees=100)
    classifier.fit(x_train, y_train)

    distribution = classifier.predict_proba(x_test)
    predictions = utilities.predict_from_distribution(distribution, classifier.rand, classifier.label_encoder)
    acc = utilities.accuracy(y_test, predictions)
    print(dataset_loader.__name__, seed, acc)

def experiment_list():
    for seed in range(0, 9):
        for dataset_name in ['BeetleFly', 'GunPoint', 'ECG200', 'Wine']:
            yield seed, dataset_name

if __name__ == "__main__":
    datasets_dir = '/scratch/datasets/'
    results_dir = '/scratch/results/'
    parallel = Parallel(n_jobs=-1)
    parallel(delayed(run_experiment)(datasets_dir, results_dir, 'pf', dataset_name, resampleID=seed, train_file=False, overwrite=True)
             for seed, dataset_name in experiment_list())
