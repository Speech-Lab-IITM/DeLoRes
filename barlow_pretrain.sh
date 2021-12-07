#! /bin/bash
#SBATCH --nodes=1
#SBATCH --partition=nltmp
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:A100-SXM4:2
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
srun python main_downstream_byol_gen.py \
            --epochs 100 \
            --batch_size 64 \
            --down_stream_task "musical_instruments" \
            --tag "byol_finetune_musical_instruments_complete" \
	    --freeze false \
	    --pretrain_path "/nlsasfs/home/nltm-pilot/ashishs/byol_model/BARLOW-A/checkpoints_upstream_3_byol_byol/checkpoints_deepcluster/checkpoint_94_.pth.tar" \
            --use_model byol \
	        --norm byol

