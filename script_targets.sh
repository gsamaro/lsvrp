#PBS -N targets
#PBS -q paralela
#PBS -l nodes=2:ppn=128
#PBS -e outputs/erros_targets_paralela
#PBS -o outputs/saidas_targets_paralela
#PBS -m abe
#PBS -M g229780@dac.unicamp.br

module load mpich/4.1.1-gcc-9.4.0

cd $PBS_O_WORKDIR

# Workers: workers.num is "auto", so the pool takes PBS_NP (2 x 128 = 256) and never
# more than one task per instance.
# Generates out/target/targets.xlsx from the target_generation block of config/config.json.
# A previous out/target is moved to out/target.stale-<timestamp>; an interrupted job
# resubmitted with the same configuration continues from out/target/work/.
mpirun poetry run python -m mpi4py.futures generate_targets.py
