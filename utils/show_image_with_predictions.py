import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_image_with_predictions(image, predictions):
    """Display the image with bounding boxes filtered according to the score."""
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    for _, pred in predictions.iterrows():  # Use iterrows() to iterate over DataFrame rows
        # Access the prediction values directly
        x_min = pred["x_min"]
        x_max = pred["x_max"]
        y_min = pred["y_min"]
        y_max = pred["y_max"]
        class_id = pred["class_id"]
        score = pred["score"]

        # Calculate width and height for the rectangle
        width, height = x_max - x_min, y_max - y_min

        # Create and add the rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Place the text (class ID and score) on the image
        plt.text(x_min, y_min - 10, f"{class_id}: {score:.2f}", color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()
