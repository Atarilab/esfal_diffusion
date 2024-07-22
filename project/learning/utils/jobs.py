#!/usr/bin/python
#
import os
import sys

try:
    from config import Config
    from log_manager import LogManager
except:
    from utils.config import Config
    from utils.log_manager import LogManager

# DEFAULT CONFIG
DEFAULT_JOB_NAME = "myjob"
DEFAULT_PARTITION = "lrz-dgx-1-p100x8"
DEFAULT_TIME = "8:00:00"
DEFAULT_MAIL_USER = "victor.dhedin@tum.de"

def get_arg_value(argv:list, arg_name:str):
    for i, arg in enumerate(argv):
        if arg == arg_name:
            return argv[i+1]
    return None

def pop_arg_value(argv:list, arg_name:str, default=None):
    for i, arg in enumerate(argv):
        if arg == arg_name:
            argv.pop(i)
            return argv.pop(i)
    return default

def submit_job(job):
    with open(f"job.sbatch", 'w') as fp:
        fp.write(job)
    os.system(f"sbatch job.sbatch")
    os.remove(f"job.sbatch")

def makejob(argv):
    job_name = pop_arg_value(argv, "--job-name", DEFAULT_JOB_NAME)
    partition = pop_arg_value(argv, "--partition", DEFAULT_PARTITION)
    time = pop_arg_value(argv, "--time", DEFAULT_TIME)
    mail_user = pop_arg_value(argv, "--mail-user", DEFAULT_MAIL_USER)

    cfg_path = get_arg_value(argv, "--cfg")
    assert cfg_path != None, "Can't find --cfg in " + " ".join(argv)
    cfg = Config(cfg_path)
    logdir = cfg.get_value("logdir")
    log_manager = LogManager(logdir)
    run_dir = log_manager.run_dir
    cfg.change_value("logdir", run_dir) # Otherwise it creates 2 logdir

    command_line = " ".join(argv[1:])
    print("Executing:", command_line)
    print("Job name:", job_name)
    print("Run logs directory:", run_dir)
    print()

    return f"""#!/bin/bash 

#SBATCH --job-name=ESFAL
#SBATCH --gres=gpu:1
#SBATCH --nodes=1-1
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --output={run_dir}/slurm-%A_%a.out
#SBATCH --error={run_dir}/slurm-%A_%a.err
#SBATCH --mail-user={mail_user}
#SBATCH --mail-type=end

cd

srun --container-mounts=./Workspace:/workspace \
     --container-workdir=/workspace/T4DL \
     --container-image=$PWD'/my_pytorch_jobs.sqsh' \
     {command_line}
"""

if __name__ == "__main__":
    submit_job(makejob(sys.argv))
