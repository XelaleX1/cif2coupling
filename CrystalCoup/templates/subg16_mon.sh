#!/bin/bash -l

#SBATCH -p lowpriority,nodes
#SBATCH -c 10
#SBATCH --mem=24GB
#SBATCH -t 03:00:00
#SBATCH --array 604-999
#SBATCH -J ZR1_Y6_%a
#SBATCH --output ZR1_Y6_%a.out

module load apps/gaussian/16
export GAUSS_SCRDIR=/mnt/lustre/users/$USER/g16scratch/$SLURM_JOB_ID
export InpDir=`pwd`
export WorkDir=/tmp/$SLURM_JOB_ID

mkdir -p $GAUSS_SCRDIR
. $g16root/g16/bsd/g16.profile

i=$((SLURM_ARRAY_TASK_ID))
printf -v j "%04d" $i

mkdir -p ${WorkDir}
cp ${j}.com ${WorkDir}
cd ${WorkDir}

chk=`grep chk ${j}.com | cut -d "=" -f 2`

g16 ${j}.com ${InpDir}/${j}.log

formchk -3 ${chk}
rm ${chk}

cp ${chk%.*}.fchk ${InpDir}
cd ${InpDir}

rm -rf $GAUSS_SCRDIR ${WorkDir}
