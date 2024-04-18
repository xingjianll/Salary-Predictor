from typing import List, Dict, Any

from torch import nn
from torch.utils.data import Dataset
from transformers import OpenAIGPTModel
import torch


class GPT1Dataset(Dataset):
    def __init__(self, input_ids, attention_mask, categorical_features, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.categorical_features = categorical_features
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'categorical_features': self.categorical_features[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

    def __len__(self):
        return len(self.labels)


class GPT1(nn.Module):
    def __init__(self, num_categorical_features):
        super(GPT1, self).__init__()
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype="float16",
        #     bnb_4bit_use_double_quant=False,
        # )
        #
        # device_map = {"": 0}

        self.gpt = OpenAIGPTModel.from_pretrained("openai-gpt",
                                                  # quantization_config=bnb_config,
                                                  # device_map=device_map
                                                  )

        self.fc1 = nn.Linear(self.gpt.config.hidden_size + num_categorical_features, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(100, 1)  # Output layer for salary prediction

    def forward(self, input_ids, attention_mask, categorical_features):
        # Process textual input through GPT
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)

        text_features = outputs.last_hidden_state[:, -1, :]  # Use the last token's representation

        # Concatenate text features with categorical features
        combined_features = torch.cat((text_features, categorical_features), dim=1)
        combined_features = self.fc1(combined_features)

        x = self.relu(combined_features)
        x = self.dropout(x)
        x = self.output(x)

        return x.squeeze() 

    def prepare_inputs_for_generation(self, haha1: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to prepare inputs in the
        generate method.
        """
        return {"haha2": haha1}


def collate_batch(batch):
    """Custom collate function for handling batches of data where all input tensors are of the same length."""

    # Separate and stack the data directly since all tensors are already of the same length
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    categorical_features = torch.stack([item['categorical_features'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'categorical_features': categorical_features,
    }
