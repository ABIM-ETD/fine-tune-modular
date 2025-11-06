import unsloth
from unsloth import FastModel
import torch




def get_base_model(model_name):
    model,tokenizer=FastModel.from_pretrained(
        model_name=model_name,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
        max_seq_length=2048
    )
    return model,tokenizer



def setting_up_model(model_name):


    base_model,tokenizer=get_base_model(model_name=model_name)
    peft_model = FastModel.get_peft_model(
    base_model,
    finetune_vision_layers= False,
    finetune_language_layers=True,
    finetune_attention_modules= True,
    finetune_mlp_modules=True,

    r=8,
    lora_alpha=8,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout =0,
    use_gradient_checkpointing="unsloth",
    use_rslora=False,
    loftq_config=None,
    bias="none",
    random_state=3407
)
    
    return peft_model,tokenizer

