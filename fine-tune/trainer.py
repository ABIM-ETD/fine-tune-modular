from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only



def trainer(peft_model,tokenizer,final_dataset):
    trainer = SFTTrainer(
        model = peft_model,
        tokenizer = tokenizer,
        train_dataset = final_dataset,
        eval_dataset = None, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 20,
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use this for WandB etc
        ),
    )


    trainer=train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n"
    )

    return trainer