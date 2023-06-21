from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset,load_dataset,  DatasetDict
import tools

LOADFILTERED = True

def filter_streaming_dataset(dataset, filters):
  filtered_dict = defaultdict(list)
  total = 0
  for sample in tqdm(iter(dataset)):
    total += 1
    if tools.any_keyword_in_string(sample["content"], filters):
      for k, v in sample.items():
        filtered_dict[k].append(v)
  print(f"{len(filtered_dict['content'])/total:.2%} of data after filtering.")
  return Dataset.from_dict(filtered_dict)

def select_dataset(dataset, n_samples=10000):
  new_dict = defaultdict(list)
  total = 0
  for sample in tqdm(iter(dataset)):
    total += 1
    for k, v in sample.items():
      new_dict[k].append(v)
    if total>=n_samples:
      break

  return Dataset.from_dict(new_dict)

if LOADFILTERED:
  ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train", streaming=True)
  ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")
  # Select a subset of the data
  # ds_train = ds_train.select(range(10000))  # For example, first 10,000 samples
  # ds_valid = ds_valid.select(range(1000))  # For example, first 1,000 samples



  raw_datasets = DatasetDict(
      {
          "train": select_dataset(ds_train,10000),  # .shuffle().select(range(50000)),
          "valid": select_dataset(ds_valid,1000),  # .shuffle().select(range(500))
      }
  )

else:
  split = "train"  # "valid"
  filters = ["pandas", "sklearn", "matplotlib", "seaborn"]

  data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
  filtered_data = filter_streaming_dataset(data, filters)