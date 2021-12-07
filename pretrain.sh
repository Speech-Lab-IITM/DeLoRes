#! /bin/bash
#SBATCH --nodes=1
#SBATCH --partition=nltmp
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:A100-SXM4:4
#SBATCH --time=6-23:00:00
#SBATCH --error=job.%J.err
##SBATCH --output=job.%J.out
#cd $SLURM_SUBMIT_DIR
#cd /nlsasfs/home/sysadmin/nazgul/gpu-burn-master
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"
echo "Job submission directory is : $SLURM_SUBMIT_DIR"
#srun ./gpu_burn -tc -d 3600 #
#srun /bin/hostname
eval "$(conda shell.bash hook)"
conda activate barlow_final
srun python /nlsasfs/home/nltm-pilot/ashishs/byol_model/BARLOW-A/upstream_3/main.py \
    --input /nlsasfs/home/nltm-pilot/ashishs/pretrain_data_final.csv \
    --epochs 200 \
    --batch_size 1024 \
    --save_dir /nlsasfs/home/nltm-pilot/ashishs/byol_model/BARLOW-A/checkpoints_upstream_3_byol_byol \
    --num_workers 2 \
    --use_model byol \
    --final_units 8192 \
    --use_norm byol \
    --length_wave 0.96 \
    --resume \
    --checkpoint_path /nlsasfs/home/nltm-pilot/ashishs/byol_model/BARLOW-A/checkpoints_upstream_3_byol_byol/checkpoints_deepcluster/checkpoint_45_.pth.tar



