from torch.utils.data import Dataset

class PromptCompletionDataset(Dataset):
    def __init__(self, pref_dataset):
        self.pref_dataset = pref_dataset

    def __len__(self):
        return len(self.pref_dataset)

    def __getitem__(self, idx):
        return {
            "prompt": self.pref_dataset[idx]["prompt"],
            "chosen": self.pref_dataset[idx]["chosen"],
            "rejected": self.pref_dataset[idx]["rejected"],
            "prompt_id": self.pref_dataset[idx]["prompt_id"]
        }

class PromptDataset(Dataset):
    def __init__(self, pref_dataset):
        self.pref_dataset = pref_dataset

    def __len__(self):
        return len(self.pref_dataset)

    def __getitem__(self, idx):
        return {
            "prompt": self.pref_dataset[idx]["prompt"],
            "prompt_id": self.pref_dataset[idx]["prompt_id"]
        }
