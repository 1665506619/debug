MODEL_PATH=${1:-"1202_v1_egobjects_lora/checkpoint-7673"}

bash evaluation/scripts/eval_roborefit.sh $MODEL_PATH
bash evaluation/scripts/eval_rynnec.sh  $MODEL_PATH