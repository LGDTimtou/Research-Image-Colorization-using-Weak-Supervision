import matplotlib.pyplot as plt
import pandas as pd
import json

output_folder = "parsed_output/"

# Load data (replace filenames with actual paths)
files_10_2 = {
    "Gaussian": f"{output_folder}GAUSSIAN_10_2.json",
    "Grid": f"{output_folder}GRID_10_2.json",
    "Poisson": f"{output_folder}POISSON_10_2.json",
    "Random": f"{output_folder}RANDOM_10_2.json"
}

bad_p_file = f"{output_folder}GAUSSIAN_20_2.json"

def train_and_val_loss_ind():
    for name, file in files_10_2.items():
        with open(file, 'r') as f:
            data = json.load(f)
        
        epochs = [epoch[0] for epoch in data['epoch_data']]
        train_loss = [epoch[1] for epoch in data['epoch_data']]
        val_loss = [epoch[2] for epoch in data['epoch_data']]
        
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss Trends Over Epochs - {name}")
        plt.legend()
        plt.grid()
        plt.show()

def average_loss_accross_distributions():
    results = []

    for name, file in files_10_2.items():
        with open(file, 'r') as f:
            data = json.load(f)
        
        for result in data['test_results']:
            results.append({
                "Distribution": result['distribution'],
                "n": result['n'],
                "p": result['p'],
                "Average Loss": result['average_loss']
            })

    df = pd.DataFrame(results)
    print(df)
    pivot_df = df.pivot_table(index=['n', 'p'], columns='Distribution', values='Average Loss')

    pivot_df.plot(kind='bar', figsize=(10, 6))
    plt.title("Average Loss Across Distributions")
    plt.xlabel("(n, p)")
    plt.ylabel("Average Loss")
    plt.xticks(rotation=45)
    plt.legend(title="Tested Distribution")
    plt.grid(axis='y')
    plt.show()


def comparison_trained_distributions():
    for name, file in files_10_2.items():
        with open(file, 'r') as f:
            data = json.load(f)
        
        results = data['test_results']
        losses = [res['average_loss'] for res in results]
        labels = [f"{res['distribution']} (n={res['n']}, p={res['p']})" for res in results]
        
        plt.figure(figsize=(12, 6))
        plt.bar(labels, losses, label=name)
        plt.title(f"Performance Comparison of Trained Distribution - {name}")
        plt.xlabel("Tested Distributions")
        plt.ylabel("Average Loss")
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()


def bad_p_showcase():
    with open(bad_p_file, 'r') as file:
        data = json.load(file)

    # Convert test results to DataFrame
    test_results = pd.DataFrame(data["test_results"])

    # Filter for Gaussian distribution and n in [5, 30]
    gaussian_data = test_results[(test_results["distribution"] == "GAUSSIAN") & (test_results["n"] >= 5) & (test_results["n"] <= 30)]

    # Plot data
    plt.figure(figsize=(12, 6))

    for n in sorted(gaussian_data['n'].unique()):
        subset = gaussian_data[gaussian_data['n'] == n]
        plt.plot(
            subset['p'], 
            subset['average_loss'], 
            marker='o', 
            label=f'n={n}', 
            linewidth=2
        )
        # Highlight p=1 and p=5
        p1_loss = subset[subset['p'] == 1]['average_loss'].values[0]
        p5_loss = subset[subset['p'] == 5]['average_loss'].values[0]
        plt.scatter([1, 5], [p1_loss, p5_loss], color='red', zorder=5, s=100)

    # Labels and legend
    plt.title("Average Loss for Gaussian Distribution (n=5 to n=30)", fontsize=16)
    plt.xlabel("p Values", fontsize=14)
    plt.ylabel("Average Loss", fontsize=14)
    plt.xticks(range(1, 6))
    plt.legend(title="n Values")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add annotation for high-loss points
    plt.text(1, gaussian_data[gaussian_data['p'] == 1]['average_loss'].max(), "High Loss\nat p=1", color="red", fontsize=12, ha="center")
    plt.text(5, gaussian_data[gaussian_data['p'] == 5]['average_loss'].max(), "High Loss\nat p=5", color="red", fontsize=12, ha="center")

    plt.tight_layout()
    plt.show()

average_loss_accross_distributions()