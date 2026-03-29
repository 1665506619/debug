MODEL_PATH=${1:-"1202_v1_egobjects_lora/checkpoint-7673"}

bash evaluation/scripts/eval_grefcoco.sh $MODEL_PATH
bash evaluation/scripts/eval_reason_vos.sh $MODEL_PATH
