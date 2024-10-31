import torch
def create_sequences(data, seq_length):
    sequences = []
    targets = []

    for i in range(len(data)-seq_length):
        seq = data[i:i + seq_length] #values defined by the seq_length
        tar = data[i+seq_length] #values right after seq ending
        sequences.append(seq)
        targets.append(tar)

    return torch.stack(sequences), torch.stack(targets)