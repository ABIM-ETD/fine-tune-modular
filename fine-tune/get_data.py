from datasets import load_dataset



def load_dataset(dataset_hf_path):
    train_dataset=load_dataset(dataset_hf_path,split="train")
    return train_dataset



def format_prompt(item):
  return {"text": f"<start_of_turn>user\n {item['conversation']}<end_of_turn><start_of_turn>model\n {item['conversation']}"}


def format_dataset(train_dataset):
   
   formatted_data=[format_prompt(item) for item in train_dataset]
   return formatted_data



    