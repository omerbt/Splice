bsub -q waic-short -R rusage[mem=60000] -R affinity[thread*2] -gpu num=1:j_exclusive=yes "source /etc/profile.d/modules.sh;module load anaconda/3.7.3; source activate /home/labs/leeat/omerba/.conda/envs/omerpy3.8; python3 -u train.py dataset.direction=BtoA dataset.global_crops.n_crops=1 dataset.global_crops.min_cover=1 loss.lambda_global_ssim=0 loss.lambda_patch_ssim=0 loss.lambda_identity=0 dataset.dataroot=./datasets/horse_zebra"
