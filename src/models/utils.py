import csv
from typing import List, Tuple, Any

import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator
from typing import List, Tuple, Any


def read_csv_data(file_location: str, input_columns: List[str], target_column: str) -> List[Tuple[List[Any], Any]]:
    """
    Reads a CSV file and extracts specified input columns and a target column.

    Parameters:
        file_location (str): The path to the CSV file.
        input_columns (List[str]): List of column names to be used as inputs.
        target_column (str): Column name to be used as the target.

    Returns:
        List[Tuple[List[Any], Any]]: Each tuple contains a list of input values and the target value.
    """
    data = []
    with open(file_location, mode='r', newline='') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            inputs = [row[col] for col in input_columns if col in row]
            target = row[target_column] if target_column in row else None
            data.append((inputs, target))
    return data


def clean_location(data: List[Tuple[List[Any], Any]], location_index: int) -> List[Tuple[List[Any], Any]]:
    """
    Cleans the location data in the list of tuples to retain only the state abbreviation.

    Parameters:
        data (List[Tuple[List[Any], Any]]): The data containing tuples of inputs and a target value.
        location_index (int): The index of the location data in the input list.

    Returns:
        List[Tuple[List[Any], Any]]: The cleaned data with only the state abbreviation in the location.
    """
    cleaned_data = []
    for inputs, target in data:
        # Split the location string by comma and strip any leading/trailing spaces
        if len(inputs) > location_index and ',' in inputs[location_index]:
            items = inputs[location_index].split(',')

            inputs[location_index] = ''
            for item in items:
                if len(item.strip()) == 2:
                    inputs[location_index] = item.strip()
                    break
        else:
            inputs[location_index] = ''

        cleaned_data.append((inputs, target))
    return cleaned_data


def build_column_vocabulary(data: List[Tuple[List[str], Any]], column_index: int, min_freq: int = 1, specials: List[str] = ['<bos>', '<eos>', '<unk>', '<pad>']) -> Vocab:
    """
    Builds a vocabulary for a specific column in the dataset, treating empty strings or None as <unk>.

    Parameters:
        data (List[Tuple[List[str], Any]]): The dataset, where each item is a tuple containing a list of inputs and a target value.
        column_index (int): The index of the column for which to build the vocabulary.
        min_freq (int): Minimum frequency for a word to be included in the vocabulary.
        specials (List[str]): Special tokens to include in the vocabulary.

    Returns:
        Vocab: The vocabulary for the specified column.
    """
    # Tokenize the data for the specific column
    tokens = []
    for inputs, _ in data:
        if len(inputs) > column_index:
            column_value = inputs[column_index]
            # Treat empty strings or None as unknown
            if column_value is None or column_value.strip() == '':
                tokens.append('<unk>')
            else:
                tokens.append(column_value)
        else:
            raise ValueError("column_index exceeds input size")

    # Build the vocabulary for the column
    vocab = build_vocab_from_iterator([tokens], specials=specials, min_freq=min_freq)
    vocab.set_default_index(vocab['<unk>'])  # Set the default index for unknown words

    return vocab



def accuracy(model, dataset: list[tuple]):
    """
    copied from csc413 lab 1
    Compute the accuracy of `model` over the `dataset`.
    We will take the **most probable class**
    as the class predicted by the model.

    Parameters:
        `model` - A torch.nn model. We will only be passing `nn.Linear` models.
                  However, to make your code more generally useful, do not access
                  `model.weight` and `model.bias` parameters directly. These
                  class attributes may not exist for other kinds of models.
        `dataset` - A list of 2-tuples of the form (x, t), where `x` is a PyTorch
                  tensor of shape [1, 28, 28] representing an MNIST image,
                  and `t` is the corresponding target label

    Returns: a floating-point value between 0 and 1.
    """
    total = 0
    distance = 0
    for x, t in dataset:
        z = model(x)
        distance += abs(t-z)
        total += 1

    return distance / total
