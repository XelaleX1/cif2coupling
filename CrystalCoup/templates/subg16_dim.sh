#!/bin/bash -l

#SBATCH -p lowpriority,nodes
#SBATCH -c 10
#SBATCH --mem=24GB
#SBATCH -t 03:00:00
#SBATCH --array 1-603
#SBATCH -J L8-BO
#SBATCH --output L8-BO_%a.out

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
rwf=`grep rwf ${j}.com | cut -d "=" -f 2`

g16 ${j}.com ${InpDir}/${j}.log

rwfdump ${rwf} ${rwf%.*}.tmp.dat 536r
sed -n '/Dump of file/,$p' ${rwf%.*}.tmp.dat > ${rwf%.*}.fmat.dat

cp ${rwf%.*}.fmat.dat ${InpDir}
cd ${InpDir}

rm -rf $GAUSS_SCRDIR ${WorkDir}
