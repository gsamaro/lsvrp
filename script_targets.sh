#PBS -N targets
#PBS -q parexp
#PBS -l nodes=2:ppn=48
#PBS -e outputs/erros_targets
#PBS -o outputs/saidas_targets
#PBS -m abe
#PBS -M g229780@dac.unicamp.br

module load mpich/4.1.1-gcc-9.4.0

cd $PBS_O_WORKDIR

# Generates out/target/targets.xlsx from the target_generation block of config/config.json.
# A previous out/target is moved to out/target.stale-<timestamp>; an interrupted job
# resubmitted with the same configuration continues from out/target/work/.
mpirun poetry run python -m mpi4py.futures generate_targets.py
