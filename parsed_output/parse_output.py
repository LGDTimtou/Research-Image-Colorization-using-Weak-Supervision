import re
import sys
import json
import os

def parse_output(data):
    # Extract the distribution, n, and p values
    trained_distribution = re.search(r"distribution: (.+)", data).group(1).strip().replace('SamplingOption.', "")
    trained_n = int(re.search(r"n: (\d+)", data).group(1))
    trained_p = int(re.search(r"p: (\d+)", data).group(1))

    # Extract train and validation losses
    epoch_data = []
    epoch_pattern = r"Epoch \[(\d+)/\d+\], Train Loss: ([\d\.]+), Validation Loss: ([\d\.]+)"
    for match in re.finditer(epoch_pattern, data):
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        val_loss = float(match.group(3))
        epoch_data.append((epoch, train_loss, val_loss))

    test_results_pattern = re.compile(
        r"- distribution: (SamplingOption\.\w+)\s+- n: (\d+)\s+- p: (\d+)\s+- average regular loss: ([\d\.]+)\s+- average ciede loss: ([\d\.]+)"
    )
    test_results = []
    for match in test_results_pattern.finditer(data):
        distribution = match.group(1).replace('SamplingOption.', "")
        n = int(match.group(2))
        p = int(match.group(3))
        avg_regular_loss = float(match.group(4))
        test_results.append({
            "distribution": distribution,
            "n": n,
            "p": p,
            "average_loss": avg_regular_loss,
        })

    output = {
        "trained_distribution": trained_distribution,
        "trained_n_value": n,
        "trained_p_value": p,
        "epoch_data": epoch_data,
        "test_results": test_results
    }

    filename = f"{trained_distribution}_{trained_n}_{trained_p}"

    return output, filename

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        with open(file_path, 'r') as file:
            data = file.read()
        output, filename = parse_output(data)

        with open(f"parsed_output/{filename}.json", "w") as f:
            json.dump(output, f, indent=2)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
