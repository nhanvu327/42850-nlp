import csv
from rouge import Rouge

# Initialize the Rouge object
rouge = Rouge()

# Read the CSV file
with open('results/test-t5.csv', 'r') as file:
    csv_reader = csv.DictReader(file)

    # Initialize variables to store the cumulative scores
    total_scores = {'rouge-1': {'f': 0, 'p': 0, 'r': 0},
                    'rouge-2': {'f': 0, 'p': 0, 'r': 0},
                    'rouge-l': {'f': 0, 'p': 0, 'r': 0}}
    num_samples = 0

    # Iterate over each row in the CSV
    for row in csv_reader:
        predicted_summary = row['summary']
        reference_summary = row['highlights']

        # Calculate ROUGE scores for the current pair
        scores = rouge.get_scores(predicted_summary, reference_summary)

        # Accumulate the scores
        for metric in ['rouge-1', 'rouge-2', 'rouge-l']:
            for score_type in ['f', 'p', 'r']:
                total_scores[metric][score_type] += scores[0][metric][score_type]

        num_samples += 1

    # Calculate the average scores
    avg_scores = {}
    for metric, scores in total_scores.items():
        avg_scores[metric] = {score_type: score / num_samples for score_type, score in scores.items()}

    # Print the average scores
    print(avg_scores)