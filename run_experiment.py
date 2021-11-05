"""Sample script to launch a set of training runs on SLURM.

  Usage:

    $ python run_experiment.py /path/to/experiment/configs --debug

  Options:

    --debug(Boolean): Only train for 5 epochs on limited data, and delete temp logs
"""

import argparse
import os
import subprocess

from interlacer import *

parser = argparse.ArgumentParser(
    description='Run experiment to train several models specified in a list of config files.')
parser.add_argument(
    'experiment',
    help='Path to .txt experiment file or directory containing multiple configs.')
parser.add_argument(
    '--debug',
    help='Boolean indicating whether to run small-scale training experiment.',
    action='store_true')

args = parser.parse_args()
experiment_path = args.experiment
debug = args.debug

experiment = experiment_path.split('/')[-1]
for exp_config in [i for i in os.listdir(
        experiment_path) if 'ipynb' not in i]:
    this_config = os.path.join(experiment_path, exp_config)
    exp_config_name = exp_config[:-4]
    log_path = os.path.join(
        experiment_path.replace(
            'configs/',
            'training/'),
        exp_config_name +
        '/log.txt')

    command = 'srun -p rtx8000 --mem-per-gpu 45 --gpus 1 --nodes 1  --job-name ' \
            + experiment + ' --time=7-00:00:00 --account=lcnrtx python train.py ' + \
        this_config + ' --experiment ' + experiment
    if(debug):
        command += ' --debug'

    if(not os.path.exists(os.path.dirname(log_path))):
        os.makedirs(os.path.dirname(log_path))

    command += ' > ' + log_path
    subprocess.Popen([command], shell=True)
