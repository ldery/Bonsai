# Bonsai [Still Under Construction]
Code for ["Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes"](https://arxiv.org/abs/2402.05406)

## Installation
See INSTALLATION.MD for instructions on installing the appropriate libraries

## Before running pruning
Since we are performing structured pruning, we need to modify the codebase of the model to be pruned to allow us to create virtual sub-models. `lib/modelling_llama_mod.py` contains an example of this.
The parts of the model that are modified from the original are indicated by ``Bonsai Modification [Starts/Ends]`` tags to make it clear the changes made to the base huggingface modelling_llama.py codebase.

For pruning with respect to a bespoke evaluation metric, make sure to modify `lib/eval.py` with a new evaluation function and update `main.py` to use the updated evaluation function. For our experiments, we focus on
pruning with respect to perplexity. Note that the metric does not have to be differentiable since we do not perform gradient based optimization.

## Run pruning
CUDA_VISIBLE_DEVICES=0 python main.py \
  --wandb_project_name {{NAME OF PROJECT FOR WANDB}} \
  --masks_per_iter {{NUMBER OF SUB-MODELS TO EVALUATE PER ROUND}} \
  --nsamples {{NUMBER OF CALIBERATION SAMPLES PER ROUND}}  \
  --sparsity_ratio {{TARGET SPARSITY TO PRUNE TO}}  \
  --save {{WHERE TO SAVE OUTPUTS}}   \
  --prune_frac {{FRACTION PRUNED PER ROUND}}  \
  --bsz {{INSTANTANEOUS BATCH SIZE FOR FORWARD PASS}} # Default to 1 \
  --prune_method {{METHOD USED TO DEFINE PRIOR}} # Default to wanda  \
  --dataset {{DATASET TO PRUNE WITH RESPECT TO}} # Default to wikitetxt  \

### Run to produce LLama-2 7B WikiText Model from Paper
`python my_main.py --model meta-llama/Llama-2-7b-hf --dataset wikitext2 --sparsity_ratio 0.5 --wandb_project_name ReprodLLama-2-Wikitext --masks_per_iter 200 --nsamples 32 --save outdir  --prune_frac 0.05 --bsz 1 --prune_method wanda`

### Run output
We do not save the whole pruned model due to space constraints. What we do save is a pickled dictionary of the pruning masks generated at each of pruning (files are saved in whatever folder is specified in `--save`)

## Post-pruning adaptation
After pruning the model, we can perform parameter efficient fine-tuning on the model to obtained an adapted model

`>> cd lora_ft`

CUDA_VISIBLE_DEVICES=0 python finetune_lm.py \
	--model_name_or_path "meta-llama/Llama-2-7b-hf" \
	--config_name "meta-llama/Llama-2-7b-hf" \
	--num_train_epochs 1 \
	--block_size 512 \ 
	--lora_r 128 \ 
	--learning_rate 1e-4 \
	--lora_alpha_ratio 4 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 8 \
	--do_train \
	--do_eval \
	--max_train_samples 15000 \
	--max_eval_samples 128 \
	--overwrite_output_dir \
	--output_dir  {{PATH TO SAVE FINAL MODEL }}  \
	--prune_info_path {{PATH WHERE PRUNING MASKS WERE SAVED }} \
	--hidden_mse_weight 0.0 \
	--kl_weight 0.01 \
	--dataset_name "wikitext" \
	--dataset_config_name "en" \



