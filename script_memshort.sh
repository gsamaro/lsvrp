#PBS -N main
#PBS -q memshort
#PBS -l nodes=1:ppn=128
#PBS -e outputs/erros_memshort
#PBS -o outputs/saidas_memshort
#PBS -m abe
#PBS -M g229780@dac.unicamp.br

module load mpich/4.1.1-gcc-9.4.0

source phd3/bin/activate

cd $PBS_O_WORKDIR

mpirun python -m mpi4py.futures main.py
