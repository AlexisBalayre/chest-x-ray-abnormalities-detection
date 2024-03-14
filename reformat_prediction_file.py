import csv

# Nom du fichier CSV d'entrée
input_csv_file = "/Users/alexis/Cranfield/AI/assignment/models/Faster-R-CNN/test_predictions.csv"
# Nom du fichier CSV de sortie
output_csv_file = "/Users/alexis/Cranfield/AI/assignment/models/Faster-R-CNN/test_predictions_reformatted.csv"

if __name__ == "__main__":
    # Ouvrir le fichier CSV d'entrée pour lire les données
    with open(input_csv_file, mode="r") as infile:
        reader = csv.reader(infile)
        # Passer l'en-tête
        next(reader, None)

        # Ouvrir le fichier CSV de sortie pour écrire les transformations
        with open(output_csv_file, mode="w", newline="") as outfile:
            writer = csv.writer(outfile)
            # Écrire l'en-tête du fichier de sortie
            writer.writerow(
                ["image_id", "class_id", "score", "x_min", "y_min", "x_max", "y_max"]
            )

            for row in reader:
                image_id, prediction_string = row
                # Vérifier si la chaîne de prédiction n'est pas vide
                if prediction_string:
                    # Découper la chaîne de prédiction en éléments individuels
                    predictions = prediction_string.split(" ")
                    # Itérer sur chaque prédiction (sachant qu'une prédiction complète comporte 6 éléments)
                    for i in range(0, len(predictions), 6):
                        class_id = predictions[i]
                        score = predictions[i + 1]
                        x_min = predictions[i + 2]
                        y_min = predictions[i + 3]
                        x_max = predictions[i + 4]
                        y_max = predictions[i + 5]
                        # Écrire la prédiction décomposée dans le fichier de sortie
                        writer.writerow(
                            [image_id, class_id, score, x_min, y_min, x_max, y_max]
                        )
