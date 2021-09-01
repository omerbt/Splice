bsub -q waic-short -R rusage[mem=60000] -R affinity[thread*2] -gpu num=1:j_exclusive=yes "source /etc/profile.d/modules.sh;module load anaconda/3.7.3; source activate /home/labs/leeat/omerba/.conda/envs/omerpy3.8; python3 -u train.py --dataroot ./datasets/single_image_horse_zebra --netG skip --cls_lambda 1 --input_noise True --lambda_identity 0 --lambda_GAN 0 --lambda_patch_ssim 0 --lambda_global_ssim 0 --skip_activation tanh"
