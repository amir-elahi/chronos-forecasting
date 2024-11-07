from scripts.training.train import ChronosDataset, ChannelDataset
from src.chronos import ChronosConfig
from gluonts.dataset.common import FileDataset
import torch
from torch.utils.data import Dataset
import h5py

datasets = ["./scripts/kernelsynth-data.arrow", "./scripts/passengers.arrow"]
probability = [1, 1]

chronos_config = ChronosConfig(
    tokenizer_class="MeanScaleUniformBins",
    tokenizer_kwargs={'low_limit': -15.0, 'high_limit': 15.0},
    n_tokens=4096,
    n_special_tokens=2,
    pad_token_id=0,
    eos_token_id=1,
    use_eos_token=True,
    model_type="seq2seq",
    context_length=512,
    prediction_length=64,
    num_samples=20,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
)


temp = [FileDataset(dataset, freq='h') for dataset in datasets]

tokenizer = chronos_config.create_tokenizer()
train_dataset = ChronosDataset(temp, probability, tokenizer)

print('#' * 10, 'Chronos Dataset', '#' * 10)

for item in train_dataset:
    for key, value in item.items():
        print(key, value.shape)
    break

# print('#' * 10, 'Channel Dataset', '#' * 10)

# datasets = [
#     ["./trainingData/contextTarget.pt", "./trainingData/context1.pt", "./trainingData/context2.pt"],
#     ["./trainingData/context1.pt", "./trainingData/context1.pt", "./trainingData/context1.pt"]
# ]

# probability = [1, 1]

# loaded_tensors = [[torch.load(path, weights_only=True) for path in inner_list] for inner_list in datasets]

# train_dataset = ChannelDataset(loaded_tensors, probability, tokenizer)

# for item in train_dataset:
#     for key, value in item.items():
#         print(key, value.shape)
#     break

print('#' * 10, 'Channel Dataset with Lazy tensor', '#' * 10)


class LazyTensorDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.num_datasets = len(f.keys())  # Only count available datasets
            print("Total datasets:", self.num_datasets)  # Debugging info

    def __len__(self):
        return self.num_datasets  # Number of datasets stored in the HDF5 file

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as f:
            dataset_name = f"dataset_{index}"
            data = f[dataset_name][()]  # Load the tensor array for the given dataset
            # Convert the array into a list of tensors
            return [torch.tensor(item) for item in data]

    def __iter__(self):
        # Define an iterator that yields only `__len__()` items
        for i in range(self.__len__()):
            yield self.__getitem__(i)


# Step 2: Instantiate the lazy dataset
datasets = "./scripts/MultivariateData.h5"

probability = [1, 0, 0, 0, 0]

lazy_tensor_dataset = LazyTensorDataset(datasets)

train_dataset = ChannelDataset(lazy_tensor_dataset, probability, tokenizer)

for item in train_dataset:
    for key, value in item.items():
        print(key, value.shape)
    # break
