from .get_model import setting_up_model
from .get_data import load_dataset,format_dataset
from .trainer import trainer



model_name='unsloth/medgemma-4b-it'
dataset_hf_path="akemiH/NoteChat"


trained_dataset=load_dataset(dataset_hf_path)

final_dataset=format_dataset(trained_dataset)

peft_model,tokenizer=setting_up_model(model_name=model_name)

trainer=trainer(peft_model=peft_model,tokenizer=tokenizer,final_dataset=final_dataset)


##Training

trainer.train()











