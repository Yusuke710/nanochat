# startup for human user on remote GPU machine

git clone -b deepseek-ocr-llava https://github.com/Yusuke710/nanochat.git
cd nanochat/
claude_yolo

copy .env
fix batchsize
change arg in WANDB_RUN= NPROC_PER_NODE= bash speedrun_vision.sh

mkdir -p reference_code
cd reference_code
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd ../nanochat


You are in a new environment. Install uv and run uv sync to install all the dependencies you need to run in this GPU environment
According to the plan, tier 1 training is complete in the previous sesion. However, we never ran training with multiple GPUs and we want to check that. Read https://github.com/karpathy/nanochat/discussions/1 for instruction ot train on multiple gpu and check if out vis_tok_train.py can do the same.


You are in a new environment, 
- below is the command to run the full pipeline, from environmental construction to stage 1 and 2 training 
export HF_TOKEN=<your-hf-token> WANDB_API_KEY=<your-wandb-key>
WANDB_RUN=testH100 NPROC_PER_NODE=1 bash speedrun_vision.sh
- Check last commit, I have added the code for gradient checkpoint. Please run vis_tok training with and without gradient checkpoint and compare which one is better as there is a tradeoff between more batchsize and slower trainining due to recomputation

please tell me
- any error you encountered and how you resolved it.
- if you encounter error, run each command inside speedrun_vision.sh and keep going
- tell me the max batch size that fits in H100 NVL(93GB) for both stage 1 and 2
- other metrics worth noting
- make sure you update findings.md and reproduce.md as you go 

tep 00020/1806 (01.16%) | loss: 4.7117 | lr: 5.00e-05 | dt: 10843.04ms | tok/sec: 1,511 | mfu: 0.63 | total time: 2.71m  

