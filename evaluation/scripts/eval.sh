GIT_SSH_COMMAND="ssh -i /lustre/fs11/portfolios/llmservice/projects/llmservice_nlp_fm/users/zhidingy/wsh-ws/playground/region/code/sshkey" git pull origin sam3
MODEL_PATH=${1:-"0106_pretrain_v2_new_null_format_fix_lora/checkpoint-11000"}

bash evaluation/scripts/eval_refcoco.sh $MODEL_PATH
bash evaluation/scripts/eval_grefcoco.sh $MODEL_PATH