from trl import GRPOConfig, GRPOTrainer
from datasets import load_from_disk
from transformers import AutoModelForCausalLM
from peft import LoraConfig
from utils.print_utils import warn, notice, log_to_file
from reviewer.judge import LLMJudger


REWARD_SCALE = 5
rewards_memory = []


@log_to_file("./grpo.log")
def grpo(venue: str) -> None:
    sft_path = f"./model/{venue}/Qwen2.5-7B-Instruct-sft"
    grpo_path = f"./model/{venue}/Qwen2.5-7B-Instruct-grpo"
    dataset_path = f"./data/{venue}/comparisons_grpo"

    dataset = load_from_disk(dataset_path)
    dataset = dataset["train"]

                      
    def dblp_reward_func(completions, ground_truth, **kwargs):
              
        rewards = []
        predicted_answers = []
        for completion, gt in zip(completions, ground_truth):
                  
            pred_ans = LLMJudger.extract_answer(completion)
            predicted_answers.append(pred_ans)
            reward = int(pred_ans == int(float(gt))) * REWARD_SCALE
            rewards.append(reward)
              
        i = 0
        for pa, r in zip(predicted_answers, rewards):
            if r > 0:
                warn(pa, end=" ")
            else:
                warn(pa, color="red", end=" ")
            i += 1
            if i % 8 == 0:
                print("-", end=" ")
        print()
        rewards_memory.append(rewards)
        return rewards

    model = AutoModelForCausalLM.from_pretrained(
        sft_path,
        torch_dtype="bfloat16",
    )

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
                                              
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    grpo_config = GRPOConfig(
        bf16=True,
        use_vllm=True,
        num_train_epochs=1,
        gradient_accumulation_steps=64,
        per_device_train_batch_size=4,
        learning_rate=5e-4,
        max_grad_norm=1.0,
        num_generations=8,
        max_completion_length=512,
        average_tokens_across_devices=False,
        use_liger_loss=True,                        
        epsilon=0.2,
        beta=0.0,                  
        epsilon_high=0.28,                        
        loss_type="dr_grpo",                     
        scale_rewards=False,                  
        save_strategy="steps",
        save_steps=10,
        output_dir="./results",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=dblp_reward_func,
        train_dataset=dataset,
        args=grpo_config,
        peft_config=lora_config,
    )
    trainer.train()
                     
    trainer.save_model(grpo_path)

                           
    with open('./rewards_output.txt', 'a') as f:
        f.write(str(rewards_memory) + '\n')


if __name__ == "__main__":
    grpo("iclr_2025")

