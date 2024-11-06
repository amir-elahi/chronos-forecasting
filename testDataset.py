from scripts.training.train import ChronosDataset, ChannelDataset
from src.chronos import ChronosConfig
from gluonts.dataset.common import FileDataset
import torch
from torch.utils.data import Dataset

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
    def __init__(self, datasets):
        self.datasets = datasets  # List of lists of file paths

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        # Load tensors from file paths for the specific index
        inner_list = self.datasets[index]
        return [torch.load(path, weights_only=True) for path in inner_list]


# Step 2: Instantiate the lazy dataset
datasets = [
    ["./trainingData/contextTarget.pt", "./trainingData/context1.pt", "./trainingData/context2.pt"],
    ["./trainingData/context1.pt", "./trainingData/context1.pt", "./trainingData/context1.pt"]
]

probability = [0, 1]

lazy_tensor_dataset = LazyTensorDataset(datasets)

train_dataset = ChannelDataset(lazy_tensor_dataset, probability, tokenizer)

for item in train_dataset:
    for key, value in item.items():
        print(key, value.shape)
    # break
