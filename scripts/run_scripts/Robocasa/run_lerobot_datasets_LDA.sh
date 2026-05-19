export WANDB_API_KEY=api/key # replace with your wandb api key

NNODES=${SENSECORE_PYTORCH_NNODES:-1}
NPROC_PER_NODE=${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}
NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=29500

echo "=== 分布式配置 ==="
echo "节点数：$NNODES"
echo "每节点进程数：$NPROC_PER_NODE"
echo "主节点：$MASTER_ADDR:$MASTER_PORT"

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "=== 网络调试信息 ==="
echo "MASTER: $MASTER_ADDR:$MASTER_PORT"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
# 测试主节点连通性
ping -c 2 $MASTER_ADDR


Framework_name=QwenMMDiT
base_vlm=base_vlm_path # replace with your own path
vision_encoder_path=vision_encoder_path # should be the parent path of vision encoder ckpt

# freeze_module_list='qwen_vl_interface,action_model.vision_encoder' # if you would like to directly train on the robocasa dataset, unfreeze vlm could obtain better performance
freeze_module_list='action_model.vision_encoder'
DIT_TYPE="DiT-L"

llavadata="asv2_conversation_en,asv2_detailed_description_en"
data_root_dir=data_root_dir # replace with your own path
data_mix=fourier_gr1_unified_1000_two_history # should be recorded in data_config.py
# fourier_gr1_arms_waist_twohistory_no_action_history
obs_horizon=2
state_dim=58 # if set null, will not use state
action_dim=138
max_num_embodiments=32 
use_delta_action=false
positional_embeddings=null # null, sinusoidal, rope

num_layers=16
repeated_diffusion_steps=4

future_obs_index=16
run_root_dir=robocasa # replace with your own path
run_id=run_id # change this to your own run id

pretrained_checkpoint=pretrained/model/path # set to null if training from scratch

only_policy=false
policy_and_video_gen=false
only_wo_video_gen=false
TRAINING_TASK_WEIGHTS="[1,1,1,1]"

export WANDB_MODE=online
wandb_entity=KaiLiu-Personal

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

accelerate launch \
  --config_file lda/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines $NNODES \
  --num_processes $((NNODES * NPROC_PER_NODE)) \
  --machine_rank $NODE_RANK \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  lda/training/train_LDA.py \
  --config_yaml lda/config/training/LDA_robocasa.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.vision_encoder_path ${vision_encoder_path} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.action_model.max_num_embodiments ${max_num_embodiments} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.action_model.action_dim ${action_dim} \
  --framework.action_model.obs_horizon ${obs_horizon} \
  --framework.action_model.future_obs_index ${future_obs_index} \
  --framework.action_model.only_policy ${only_policy} \
  --framework.action_model.policy_and_video_gen ${policy_and_video_gen} \
  --framework.action_model.only_wo_video_gen ${only_wo_video_gen} \
  --framework.action_model.diffusion_model_cfg.num_layers ${num_layers} \
  --framework.action_model.diffusion_model_cfg.positional_embeddings ${positional_embeddings} \
  --datasets.vla_data.use_delta_action ${use_delta_action} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.training_task_weights ${TRAINING_TASK_WEIGHTS} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 10 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 300000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.repeated_diffusion_steps ${repeated_diffusion_steps} \
  --trainer.learning_rate.base 4e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project lda-robocasa \
  --wandb_entity ${wandb_entity} \
  --is_debug False 2>&1 | tee ${output_dir}/train.log
