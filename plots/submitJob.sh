#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=30:00
#PBS -l mem=10GB
#PBS -N plot 
#PBS -M mc3784@nyu.edu
#PBS -m b -m e -m a -m abe
#PBS -j oe

module purge

SRCDIR=$HOME/NLP/MENU_PRICE_PREDICTION/plots
RUNDIR=$SCRATCH/EMB/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR

cd $PBS_O_WORKDIR
cp -R $SRCDIR/*.py $RUNDIR
cp submitJob.sh $RUNDIR
cd $RUNDIR

module load tensorflow/python2.7/20161029
module load nltk/3.0.2
module load scikit-learn/intel/0.18
python plot_emb.py 
#--word2vec "/home/mc3784/NLP/MENU_PRICE_PREDICTION/GoogleNews-vectors-negative300.bin"
