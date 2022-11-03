from os.path import abspath, splitext
from typing import List, Optional, Union

from datasets import load_dataset, logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import default_data_collator

logging.set_verbosity(logging.ERROR)


def load(
    tokenizer,
    seq_len: int,
    train_data_path: Union[str, List[str]],
    eval_data_path: Optional[str] = None,
    train_test_split: Optional[float] = None,
    worker: int = 1,
    batch_size: int = 1000,
    shuffle_seed: Optional[int] = None,
):
    def _grouping(data):
        encoded = tokenizer.batch_encode_plus(
            [tokenizer.bos_token + c + tokenizer.eos_token for c in data['content']],
            max_length=seq_len,
            truncation=True,
            return_attention_mask=False,
        )["input_ids"]

        input_ids = []
        last_input_ids = []
        last_room_no = data["room_no"][0]
        
        for ii, wr, ti in zip(encoded, data["speaker"], data["room_no"]):
            if len(last_input_ids + ii) <= seq_len + 1 and last_room_no == ti:
                last_input_ids += ii
            else:
                input_ids.append(last_input_ids)
                last_input_ids = ii
                last_room_no = ti
        data = {"input_ids" : input_ids}

        return data

    def _keymapping(data):
        input_ids = np.full((len(data["input_ids"]), seq_len), tokenizer.pad_token_id)
        attention_mask = np.zeros((len(data["input_ids"]), seq_len))
        labels = np.full((len(data["input_ids"]), seq_len), tokenizer.pad_token_id)

        for i, ids in enumerate(data["input_ids"]):
            input_ids[i, : len(ids) - 1] = ids[:-1]
            attention_mask[i, : len(ids) - 1] = np.ones((len(ids) - 1,))
            labels[i, : len(ids) - 1] = ids[1:]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_data_path = abspath(train_data_path)
    is_eval = False
    _, extention = splitext(train_data_path)

    datafiles = {"train": train_data_path}
    if eval_data_path is not None:
        assert (
            train_test_split is None
        ), "Only one of eval_data_path and train_test_split must be entered."
        datafiles["test"] = abspath(eval_data_path)
        is_eval = True

    if train_test_split is not None:
        assert (
            0.0 < train_test_split < 1.0
        ), "train_test_split must be a value between 0 and 1"
        train_test_split = int(train_test_split * 100)
        train_test_split = {
            "train": f"train[:{train_test_split}%]",
            "test": f"train[{train_test_split}%:]",
        }
        is_eval = True

    data = load_dataset(
        extention.replace(".", ""),
        data_files=datafiles,
        split=train_test_split,
    )

    data = data.map(
        _grouping,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
        remove_columns=data["train"].column_names,
    )

    data = data.map(
        _keymapping,
        batched=True,
        batch_size=batch_size,
        num_proc=worker,
    )

    if shuffle_seed is not None:
        data = data.shuffle(seed=shuffle_seed)

    return data["train"], (data["test"] if is_eval else None)