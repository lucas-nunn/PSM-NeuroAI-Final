
# tbc 
import torch

state_dict = torch.load('path/results/cnn.pth', map_location='cpu')


print(f'number of parameter tensors: {len(state_dict)}')
for name, tensor in state_dict.items():
    print(f'{name:40s} {tuple(tensor.shape)}')

total_params = sum(t.numel() for t in state_dict.values())
print(f'\ntotal parameters: {total_params:,}')
