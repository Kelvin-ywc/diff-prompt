cd //workspace/prompt/PromptDiffusion/GLIP
# flickr30k
python -m torch.distributed.launch --nproc_per_node=4 tools/finetune.py \
--config-file configs/pretrain/glip_A_Swin_T_O365.yaml  --ft-tasks configs/finetune/dif_prompt/flickr/train_v3.yaml \
--custom_shot_and_epoch_and_general_copy 0_5_1 \
--evaluate_only_best_on_test --push_both_val_and_test \
MODEL.WEIGHT //MODEL/glip/glip_a_tiny_o365.pth \
SOLVER.USE_AMP True TEST.DURING_TRAINING True TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 64 SOLVER.WEIGHT_DECAY 0.05 TEST.EVAL_TASK grounding \
MODEL.DYHEAD.USE_CHECKPOINT True SOLVER.FIND_UNUSED_PARAMETERS False \
SOLVER.TEST_WITH_INFERENCE True SOLVER.SEED 10 DATASETS.SHUFFLE_SEED 3 \
SOLVER.CHECKPOINT_PER_EPOCH 1.0 SOLVER.MODEL_EMA 0.0 SOLVER.BASE_LR 0.0001 TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 100 MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR ./result/finetune_flickr_diff_prompt_v3

# refcoco
python -m torch.distributed.launch --nproc_per_node=4 --master_port=30600 tools/finetune.py \
--config-file configs/pretrain/glip_A_Swin_T_O365.yaml  --ft-tasks configs/finetune/dif_prompt/refcoco/train_v3.yaml \
--custom_shot_and_epoch_and_general_copy 0_5_1 \
--evaluate_only_best_on_test --push_both_val_and_test \
MODEL.WEIGHT //MODEL/glip/glip_a_tiny_o365.pth \
SOLVER.USE_AMP True TEST.DURING_TRAINING True TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 64 SOLVER.WEIGHT_DECAY 0.05 TEST.EVAL_TASK grounding \
MODEL.DYHEAD.USE_CHECKPOINT True SOLVER.FIND_UNUSED_PARAMETERS False \
SOLVER.TEST_WITH_INFERENCE True SOLVER.SEED 10 DATASETS.SHUFFLE_SEED 3 \
SOLVER.CHECKPOINT_PER_EPOCH 1.0 SOLVER.MODEL_EMA 0.0 SOLVER.BASE_LR 0.0001 TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 100 MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR ./result/finetune_refcoco_diff_prompt_v3_test

python -m torch.distributed.launch --nproc_per_node=4 tools/test_grounding_net.py \
      --config-file configs/pretrain/glip_A_Swin_T_O365.yaml \
      --task_config configs/eval/refcoco/val.yaml \
      --weight result/finetune_refcoco_diff_prompt_v3/ft_task_1/model_final.pth \
      SOLVER.USE_AMP True TEST.DURING_TRAINING True TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 64 SOLVER.WEIGHT_DECAY 0.05 \
      TEST.EVAL_TASK grounding  MODEL.DYHEAD.USE_CHECKPOINT True \
      SOLVER.FIND_UNUSED_PARAMETERS False SOLVER.TEST_WITH_INFERENCE True SOLVER.USE_AUTOSTEP True \
      SOLVER.SEED 10 DATASETS.SHUFFLE_SEED 3 \
      DATASETS.DISABLE_SHUFFLE True SOLVER.CHECKPOINT_PER_EPOCH 1.0  SOLVER.MODEL_EMA 0.0 \
      TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 100 MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
      OUTPUT_DIR ./result/eval_diff_prompt_refcoco_glip_t_0921 PROMPT.ENABLE True PROMPT.TYPE Mask-VPT-V3 PROMPT.PROMPT_DEPTH 9 PROMPT.PROMPT_LENGTH 8

python -m torch.distributed.launch --nproc_per_node=4 tools/test_grounding_net.py \
      --config-file configs/pretrain/glip_A_Swin_T_O365.yaml \
      --task_config configs/eval/flickr/val.yaml \
      --weight result/finetune_flickr_diff_prompt_v3/ft_task_1/model_final.pth \
      SOLVER.USE_AMP True TEST.DURING_TRAINING True TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 64 SOLVER.WEIGHT_DECAY 0.05 \
      TEST.EVAL_TASK grounding  MODEL.DYHEAD.USE_CHECKPOINT True \
      SOLVER.FIND_UNUSED_PARAMETERS False SOLVER.TEST_WITH_INFERENCE True SOLVER.USE_AUTOSTEP True \
      SOLVER.SEED 10 DATASETS.SHUFFLE_SEED 3 \
      DATASETS.DISABLE_SHUFFLE True SOLVER.CHECKPOINT_PER_EPOCH 1.0  SOLVER.MODEL_EMA 0.0 \
      TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 100 MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
      OUTPUT_DIR ./result/eval_diff_prompt_flickr30k_0921 PROMPT.ENABLE True PROMPT.TYPE Mask-VPT-V3 PROMPT.PROMPT_DEPTH 9 PROMPT.PROMPT_LENGTH 8