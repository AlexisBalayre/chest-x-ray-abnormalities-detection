def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) between two boxes.

    Parameters:
    - boxA: The first bounding box as a list or tuple of (x_min, y_min, x_max, y_max).
    - boxB: The second bounding box as a list or tuple of (x_min, y_min, x_max, y_max).

    Returns:
    - iou: The Intersection over Union score as a float.
    """

    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])  # Maximum of the x_min coordinates
    yA = max(boxA[1], boxB[1])  # Maximum of the y_min coordinates
    xB = min(boxA[2], boxB[2])  # Minimum of the x_max coordinates
    yB = min(boxA[3], boxB[3])  # Minimum of the y_max coordinates

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(
        0, yB - yA
    )  # The area is zero if rectangles do not intersect

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])  # Area of boxA
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])  # Area of boxB

    # Compute the Intersection over Union by dividing the intersection area by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def evaluate_predictions(ground_truths, predictions, iou_threshold=0.4):
    """
    Evaluate object detection predictions against ground truths using Intersection over Union (IoU).

    Parameters:
    - ground_truths: A list of dictionaries, each representing a ground truth bounding box and its class.
    - predictions: A list of dictionaries, each representing a predicted bounding box and its class.
    - iou_threshold: The IoU threshold to determine if a prediction is considered a true positive.

    Returns:
    - A dictionary containing evaluation metrics: True Positives (TP), False Positives (FP), False Negatives (FN),
      Precision, Recall, and F1 Score.
    """

    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives

    # Keeps track of which ground truth boxes have been matched with a prediction
    matched_gt = {gt_index: False for gt_index, _ in enumerate(ground_truths)}

    for pred in predictions:
        best_iou = 0  # Best IoU for current prediction
        best_gt_index = None  # Index of the ground truth box with the best IoU

        # Compare each prediction against all ground truths
        for gt_index, gt in enumerate(ground_truths):
            # Match prediction with ground truth if they have the same image_id and class_id
            if (
                gt["image_id"] == pred["image_id"]
                and gt["class_id"] == pred["class_id"]
            ):
                iou = calculate_iou(
                    [pred["x_min"], pred["y_min"], pred["x_max"], pred["y_max"]],
                    [gt["x_min"], gt["y_min"], gt["x_max"], gt["y_max"]],
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

        # If the best IoU exceeds the threshold, consider it a true positive and mark the ground truth as matched
        if best_iou > iou_threshold:
            TP += 1
            if best_gt_index is not None:
                matched_gt[best_gt_index] = True
        else:
            FP += 1  # If no ground truth is matched above the IoU threshold, count as a false positive

    # Unmatched ground truths are considered false negatives
    FN = sum(not matched for matched in matched_gt.values())

    # Calculate precision, recall, and F1 score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
    }


def evaluate_no_finding(ground_truths, predictions, no_finding_class_id=14):
    """
    Evaluate predictions for the "No finding" class in a set of object detection or classification results.

    Parameters:
    - ground_truths: A list of dictionaries, each representing a ground truth with 'image_id' and 'class_id'.
    - predictions: A list of dictionaries, each representing a prediction with 'image_id' and 'class_id'.
    - no_finding_class_id: The class ID that represents "No finding". Defaults to 14.

    Returns:
    - A dictionary containing evaluation metrics: True Positives (TP), False Positives (FP), False Negatives (FN),
      and Accuracy.
    """

    # Initial count for correctly predicted as "No finding", incorrectly predicted as "No finding", and missed "No finding"
    TP = 0  # Correctly predicted as "No finding"
    FP = 0  # Incorrectly predicted as "No finding" or predicted something else when it was "No finding"
    FN = 0  # Missed predicting "No finding"

    # Set of image IDs where the ground truth is "No finding"
    no_finding_image_ids = {
        gt["image_id"] for gt in ground_truths if gt["class_id"] == no_finding_class_id
    }

    # Set of image IDs for which predictions have been made
    predicted_image_ids = {pred["image_id"] for pred in predictions}

    # Count TP and FP
    for pred in predictions:
        if pred["class_id"] == no_finding_class_id:
            if pred["image_id"] in no_finding_image_ids:
                TP += 1  # Correct prediction of "No finding"
            else:
                FP += 1  # Incorrect prediction as "No finding"

    # Count FN - missed "No finding"
    FN = len(no_finding_image_ids) - TP

    # Calculate accuracy
    accuracy = (TP) / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return {"TP": TP, "FP": FP, "FN": FN, "Accuracy": accuracy}
