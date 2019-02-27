import os
import json
import argparse
import config
import itertools as it
from db import db
from db import credentials
from utils import logger, py_utils
from copy import deepcopy


def package_parameters(parameter_dict, log):
    """Derive combinations of experiment parameters."""
    parameter_dict = {
        k: v for k, v in parameter_dict.iteritems() if isinstance(v, list)
    }
    keys_sorted = sorted(parameter_dict)
    values = list(it.product(*(parameter_dict[key] for key in keys_sorted)))
    combos = tuple({k: v for k, v in zip(keys_sorted, row)} for row in values)
    log.info('Derived %s combinations.' % len(combos))
    return list(combos)


def main(reset_process, initialize_db, experiment_name, remove=None, force_repeat=None):
    """Populate db with experiments to run."""
    main_config = config.Config()
    log = logger.get(os.path.join(main_config.log_dir, 'prepare_experiments'))
    if reset_process:
        db.reset_in_process()
        log.info('Reset experiment progress counter in DB.')
    if initialize_db:
        db.initialize_database()
        log.info('Initialized DB.')
    if remove is not None:
        db_config = credentials.postgresql_connection()
        with db.db(db_config) as db_conn:
            db_conn.remove_experiment(remove)
        log.info('Removed %s.' % remove)
    if experiment_name is not None:  # TODO: add capability for bayesian opt.
        if ',' in experiment_name:
            # Parse a comma-delimeted string of experiments
            experiment_name = experiment_name.split(',')
        else:
            experiment_name = [experiment_name]
        db_config = credentials.postgresql_connection()
        for exp in experiment_name:
            experiment_dict = py_utils.import_module(
                module=exp, pre_path=main_config.experiment_classes)
            experiment_dict = experiment_dict.experiment_params()
            exp_combos = package_parameters(experiment_dict, log)
            log.info('Preparing experiment.')
            assert exp_combos is not None, 'Experiment is empty.'

            # Repeat if requested
            repeats = experiment_dict.get('repeat', 0)
            if force_repeat is not None:
                repeats = force_repeat
            if repeats:
                dcs = []
                for copy in range(repeats):
                    # Need to make deep copies
                    dcs += deepcopy(exp_combos)
                exp_combos = dcs
                log.info(
                    'Expanded %sx to %s combinations.' % (
                        experiment_dict['repeat'],
                        len(exp_combos)))

            # Convert augmentations to json
            json_combos = []
            for combo in exp_combos:
                combo['train_augmentations'] = json.dumps(
                    deepcopy(combo['train_augmentations']))
                combo['val_augmentations'] = json.dumps(
                    deepcopy(combo['val_augmentations']))
                json_combos += [combo]

            # Add data to the DB
            with db.db(db_config) as db_conn:
                db_conn.populate_db(json_combos)
                db_conn.return_status('CREATE')
            log.info('Added new experiments.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset_process",
        dest="reset_process",
        action='store_true',
        help='Reset the in_process table.')
    parser.add_argument(
        "--initialize",
        dest="initialize_db",
        action='store_true',
        help='Recreate your database of experiments.')
    parser.add_argument(
        "--experiment",
        dest="experiment_name",
        default=None,
        type=str,
        help='Experiment to add to the database.')
    parser.add_argument(
        "--remove",
        dest="remove",
        default=None,
        type=str,
        help='Experiment to remove from the database.')
    parser.add_argument(
        "--repeat",
        dest="force_repeat",
        default=None,
        type=int,
        help='Force a number of repeats.')
    args = parser.parse_args()
    main(**vars(args))

