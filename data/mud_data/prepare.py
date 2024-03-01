import os
import pickle
import json
import numpy as np
from minbpe import RegexTokenizer
from sys import getsizeof

# NOTE: Grab the data
input_file_path = os.path.join(os.path.dirname(__file__), 'output.txt')
# with open(input_file_path, 'r') as f:
#     raw_data = json.load(f)

data = ''
# for key, val in raw_data.items():
#     if (val['command'] == []) or (val['context'] == []):
#         continue
#     context = '\n'.join(val['context'])
#     command = '\n'.join(val['command'])
#     data += f'<|startctx|> {context} <|endctx|> {command} '

with open(input_file_path, 'r') as file:
    data = file.read()

print(f'Size of data: {round(getsizeof(data) * 1e-6, 2)} MB')

# NOTE: Train tokenizer
vocab_size = 288
tokenizer = RegexTokenizer()
tokenizer.train(data, vocab_size=vocab_size)
tokenizer.register_special_tokens({'<|startctx|>': 288, '<|endctx|>': 289})
vocab_size += 2

# NOTE: Create the train and test splits
n = len(data)
i = data.find('<|startctx|>')
s = []
while i >= 0:
    s.append(i)
    i = data.find('<|startctx|>', i + 1)
    if i >= int(n*0.9):
        break
train_data = data[:s[-1]]
val_data = data[s[-1]:]

# NOTE: Encode both to integers
train_ids = tokenizer.encode(train_data, allowed_special='all')
val_ids = tokenizer.encode(val_data, allowed_special='all')

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

tokenizer.save('mud')
