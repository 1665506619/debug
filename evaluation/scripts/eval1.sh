GIT_SSH_COMMAND="ssh -i /lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/code/sshkey" git pull origin nv
MODEL_PATH=${1:-"1202_v1_egobjects_lora/checkpoint-7673"}

bash evaluation/scripts/eval_refcoco.sh $MODEL_PATH
# bash evaluation/scripts/eval_egomask.sh $MODEL_PATH  short
# bash evaluation/scripts/eval_egomask.sh $MODEL_PATH  medium
# bash evaluation/scripts/eval_egomask.sh $MODEL_PATH  long
