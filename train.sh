bsub -q waic-long -R rusage[mem=60000] -R affinity[thread*10] -gpu num=1:j_exclusive=yes -m hpe8k_hosts "source /etc/profile.d/modules.sh;module load anaconda/3.7.3; source activate /home/labs/leeat/omerba/.conda/envs/omerpy3.8; python3 -u train.py --dataroot ./datasets/dog_cat --netG skip --cls_lambda 1 --lambda_identity 0 --lambda_GAN 0 --lambda_patch_ssim 0.2 --lambda_global_ssim 0.1 --skip_activation tanh"
