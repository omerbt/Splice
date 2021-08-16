bsub -q waic-long -R rusage[mem=60000] -R affinity[thread*10] -gpu num=1:j_exclusive=yes -m hpe8k_hosts "source /etc/profile.d/modules.sh;module load anaconda/3.7.3; source activate /home/labs/leeat/omerba/.conda/envs/omerpy3.8; python3 -u train.py --model sincut --name single_image_brugge_venice --dataroot ./datasets/single_image_brugge_venice --display_id 0 --save_latest_freq 5000"
