#PBS -N main
#PBS -q paralela
#PBS -l nodes=2:ppn=128
#PBS -e outputs/erros_paralela
#PBS -o outputs/saidas_paralela
#PBS -m abe
#PBS -M g229780@dac.unicamp.br

module load gettext/0.21-gcc-9.4.0
module load mpich/4.1.1-gcc-9.4.0

cd $PBS_O_WORKDIR

mpirun poetry run python -m mpi4py.futures main.py
