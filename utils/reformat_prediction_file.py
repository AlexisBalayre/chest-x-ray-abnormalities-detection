import csv

# Define paths for the input and output CSV files
input_csv_file = "/Users/alexis/Cranfield/AI/assignment/data/test/test_predictions.csv"
output_csv_file = (
    "/Users/alexis/Cranfield/AI/assignment/data/test/test_predictions_reformatted.csv"
)


def reformat_predictions(input_csv, output_csv):
    """
    Reads predictions from an input CSV file, reformats them, and writes the reformatted predictions to an output CSV file.

    The input CSV file should contain rows with image IDs and prediction strings. This function parses the prediction strings,
    extracts individual predictions, and writes them to the output CSV file in a structured format.

    Parameters:
    - input_csv (str): Path to the input CSV file containing original predictions.
    - output_csv (str): Path to the output CSV file where reformatted predictions will be saved.
    """

    # Open the input CSV file for reading
    with open(input_csv, mode="r") as infile:
        reader = csv.reader(infile)

        # Skip the header row
        next(reader, None)

        # Open the output CSV file for writing
        with open(output_csv, mode="w", newline="") as outfile:
            writer = csv.writer(outfile)

            # Write the header row for the output file
            writer.writerow(
                ["image_id", "class_id", "score", "x_min", "y_min", "x_max", "y_max"]
            )

            # Iterate over each row in the input file
            for row in reader:
                image_id, prediction_string = row

                # Check if the prediction string is not empty
                if prediction_string:
                    # Split the prediction string into individual prediction components
                    predictions = prediction_string.split(" ")

                    # Iterate over the predictions in steps of 6 (since each prediction includes 6 values)
                    for i in range(0, len(predictions), 6):
                        # Extract each component of the prediction
                        class_id = predictions[i]
                        score = predictions[i + 1]
                        x_min = predictions[i + 2]
                        y_min = predictions[i + 3]
                        x_max = predictions[i + 4]
                        y_max = predictions[i + 5]

                        # Write the reformatted prediction to the output file
                        writer.writerow(
                            [image_id, class_id, score, x_min, y_min, x_max, y_max]
                        )


if __name__ == "__main__":
    # Call the reformatting function with specified input and output file paths
    reformat_predictions(input_csv_file, output_csv_file)
