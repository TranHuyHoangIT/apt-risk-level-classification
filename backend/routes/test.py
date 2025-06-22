import json

with open('../model_trained/model_config.json', 'r') as f:
    config = json.load(f)
best_params = config['best_params']
input_dim = config['input_dim']
output_dim = config['output_dim']

print("best param ", best_params)
print("input ", input_dim)
print("output ", output_dim)