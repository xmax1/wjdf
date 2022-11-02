from pathlib import Path
import string 
from simple_slurm import Slurm
from numpy.random import default_rng
rng = default_rng()

OUT_TMP_DIR = Path('./tmp')
if not OUT_TMP_DIR.exists():
    OUT_TMP_DIR.mkdir()

# puts the output files neatly. Trailing / ensures has defined directory
move_command = f'mv {OUT_TMP_DIR}/o-$SLURM_JOB_ID.out {OUT_TMP_DIR}/e-$SLURM_JOB_ID.err $out_dir' 


def gen_alphanum(n=10):
    uppers = string.ascii_uppercase
    lowers = string.ascii_lowercase
    numbers = ''.join([str(i) for i in range(10)])
    characters = uppers + lowers + numbers
    name = ''.join([rng.choice(characters, replace=True) for i in range(n)])
    return name


def run_single_slurm(
    execution_file: str ='train.py', 
    submission_name: str = 'x', 
    env: str = 'dex', 
    time_h: int = 24,
    run_dir: str = None,
    exp_name: str = None,        
    **exp_kwargs
    ):

    if run_dir is None: 
        run_dir = './experiments/junk/' # path to dump experiments
    run_dir = Path(run_dir)

    if exp_name is None:
        exp_name = gen_alphanum(7)
        
    run_dir /= exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    ### EDIT DEFAULTS HERE
    slurm = Slurm(
        mail_type='FAIL',
        partition='sm3090',
        N = 1,  # n_node
        n = 8,  # n_cpu
        time=f'0-{time_h}:00:00',
        output=f'{OUT_TMP_DIR}/o-%j.out',
        error=f'{OUT_TMP_DIR}/e-%j.err',
        gres='gpu:RTX3090:1',
        job_name=submission_name
    )

    cmd = f'python -u {execution_file} ' # -u unbuffers print statements
    for k, v in exp_kwargs.items():
        cmd += f' --{k} {str(v)} '

    print('RUNNING: \n ', cmd)

    slurm.sbatch(
        f'module purge \n \
        source ~/.bashrc \n \
        module load GCC \n \
        module load CUDA/11.4.1 \n \
        module load cuDNN/8.2.2.26-CUDA-11.4.1 \n \
        conda activate {env} \n \
        pwd \n \
        nvidia-smi \n \
        out_dir={run_dir} \n \
        {cmd} | tee $out_dir/py.out \n \
        {move_command} \n \
        date "+%B %V %T.%3N"'
    )

    

if __name__ == '__main__':
    ''' NOTES
    - Assumes running a conda env. To run something else, remove conda line from slurm.sbatch
    - Other sbatch defaults can be changed in the Slurm() class, you can use all these pieces to make functions for what you need
    - run_dir is where the results go 
    - exp_name is another tag so sweeps can go together (ie keep run_dir the same but change exp_name)
    '''
    exp_kwargs = {
        # experiment hyperparams go here and are passed in as cmd line args
    }
    
    run_single_slurm(
        execution_file  = 'run.py',  # what we are running
        submission_name = 'hp',       # slurm name (not id)
        env             = 'dex',     # conda environment 
        time_h          = 24,  
        run_dir         = '../experiments/remote_demo',
        exp_name        = 'flowers',        
        **exp_kwargs
        )