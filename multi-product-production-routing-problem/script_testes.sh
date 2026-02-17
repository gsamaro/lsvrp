#PBS -N teste
#PBS -q testes
#PBS -l nodes=1:ppn=2
#PBS -e outputs/erros_testes
#PBS -o outputs/saidas_testes
#PBS -m abe
#PBS -M g229780@dac.unicamp.br

module load mpich/4.1.1-gcc-9.4.0

source phd3/bin/activate

cd $PBS_O_WORKDIR

mpirun python -m mpi4py.futures main.py
