0. Configure environment and DB.
	- Copy `config.py.template` to `config.py` and fill in missing entries to reflect your system.
	- Do the same for `db/credentials.py`
	- run `python setup.py install`

1. Create an experiment.
	- See `experiments/nist_baseline.py` for an example experiment definition.
	- See `models/seung_unet.py` for an example model specification (included in the experiment definition).
	- See `datasets/cluttered_nist_baseline.py` for an example dataset class (included in the experiment definition).
	- Initialize the DB and load an experiment: `python build_experiments.py --experiment=nist_baseline --initialize`
        - I manually access the db with `psql cluttered_nist -h 127.0.0.1 -d cluttered_nist`.

2. Run an experiment.
	- A single job from the DB: `CUDA_VISIBLE_DEVICES=0 python run_job.py`
	- A single job without the DB: `CUDA_VISIBLE_DEVICES=0 python run_job.py --no_db --experiment=nist_baseline --model=seung_unet --train=cluttered_nist_baseline --test=cluttered_nist_baseline`
	- A local worker that continues until the DB is exhausted: `bash start_worker.sh`
	- Fill the p-nodes with workers running in Dockers: `bash docker_workers.sh`

3. Manually access the DB
        - psql cluttered_nist -h 127.0.0.1 -d cluttered_nist

4. Run and kill docker jobs
        - Run docker job `bash docker_workers.sh`
        - Kill docker jobs `python utils/docker_kill.py bash`
        - Get docker pids `docker ps`
        - Get docker job stdout `docker logs <pid>

