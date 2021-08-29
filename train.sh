bsub -q waic-long -R rusage[mem=60000] -R affinity[thread*10] -gpu num=1:j_exclusive=yes "source /etc/profile.d/modules.sh;module load anaconda/3.7.3; source activate /home/labs/leeat/omerba/.conda/envs/omerpy3.8; python3 -u train.py --dataroot ./datasets/birds --netG skip --cls_lambda 4 --dino_model_name dino_vitb8 --lambda_identity 1 --lambda_GAN 0 --lambda_patch_ssim 0.2 --lambda_global_ssim 0.1 --skip_activation tanh"
