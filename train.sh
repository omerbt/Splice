bsub -q waic-short -R rusage[mem=60000] -R affinity[thread*2] -gpu num=1:j_exclusive=yes -m hpe6k_hosts "source /etc/profile.d/modules.sh;module load anaconda/3.7.3; source activate /home/labs/leeat/omerba/.conda/envs/omerpy3.8; python3 -u train.py --dataroot ./datasets/single_image_putin_zebra --cls_lambda 10 --lambda_identity 1 --lambda_patch_ssim 0 --lambda_global_ssim 0"
