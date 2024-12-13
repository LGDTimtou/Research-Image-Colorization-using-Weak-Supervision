import os
import json
import re

# Define the directory containing the files
directory = 'final_training_output'

# Initialize a list to hold the parsed data
parsed_data = []

# Regular expressions for extracting data
regex_distribution = re.compile(r'distribution:\s*SamplingOption\.(\w+)')
regex_n = re.compile(r'n:\s*(\d+)')
regex_p = re.compile(r'p:\s*(\d+)')
regex_epoch = re.compile(r'Epoch \[(\d+)/(\d+)\], Train Loss: ([\d\.]+), Validation Loss: ([\d\.]+)')

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        
        with open(file_path, 'r') as file:
            content = file.read()
            
            # Extract metadata
            distribution_match = regex_distribution.search(content)
            n_match = regex_n.search(content)
            p_match = regex_p.search(content)

            # Parse the distribution, n, and p values
            distribution = distribution_match.group(1) if distribution_match else None
            n = int(n_match.group(1)) if n_match else None
            p = int(p_match.group(1)) if p_match else None

            # Parse epochs
            epochs = []
            for epoch_match in regex_epoch.finditer(content):
                epoch = int(epoch_match.group(1))
                train_loss = float(epoch_match.group(3))
                val_loss = float(epoch_match.group(4))
                epochs.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })

            # Append the parsed data to the list
            parsed_data.append({
                "distribution": distribution,
                "n": n,
                "p": p,
                "epochs": epochs
            })

# Save the parsed data to a JSON file
output_file = "output.json"
with open(output_file, 'w') as json_file:
    json.dump(parsed_data, json_file, indent=4)

print(f"Parsed data has been saved to {output_file}")
