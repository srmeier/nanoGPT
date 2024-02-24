import os
import pickle
import json
import numpy as np
from minbpe import RegexTokenizer
from sys import getsizeof

# NOTE: Grab the data
input_file_path = os.path.join(os.path.dirname(__file__), 'data.json')
with open(input_file_path, 'r') as f:
    raw_data = json.load(f)

data = ''
for key, val in raw_data.items():
    context = '\n'.join(val['context'])
    command = '\n'.join(val['command'])
    data += f'<|startctx|> {context} <|endctx|> {command} '

print(f'Size of data: {round(getsizeof(json.dumps(data)) * 1e-6, 2)} MB')

# NOTE: Train tokenizer
vocab_size = 288
tokenizer = RegexTokenizer()
tokenizer.train(data, vocab_size=vocab_size)
tokenizer.register_special_tokens({'<|startctx|>': 288, '<|endctx|>': 289})

# NOTE: Create the train and test splits
# TODO: Find the closest "<|startctx|>" and separate train / test
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# NOTE: Encode both to integers
train_ids = tokenizer.encode(train_data)
val_ids = tokenizer.encode(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# NOTE: Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {'vocab_size': vocab_size}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
