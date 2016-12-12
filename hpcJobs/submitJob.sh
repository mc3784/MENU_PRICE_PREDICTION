#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=8:00:00
#PBS -l mem=32GB
#PBS -N Continuous_W2V
#PBS -M mc3784@nyu.edu
#PBS -m b -m e -m a -m abe
#PBS -j oe

module purge

SRCDIR=$HOME/NLP/MENU_PRICE_PREDICTION/model/Continuous/W2V
RUNDIR=$SCRATCH/MENU/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR

cd $PBS_O_WORKDIR
cp -R $SRCDIR/*.py $RUNDIR
cp submitJob.sh $RUNDIR
cd $RUNDIR

module load tensorflow/python2.7/20161029
module load nltk/3.0.2
python train.py --data_dir "/home/mc3784/NLP/MENU_PRICE_PREDICTION/data" --word2vec "/home/mc3784/NLP/MENU_PRICE_PREDICTION/GoogleNews-vectors-negative300.bin"
