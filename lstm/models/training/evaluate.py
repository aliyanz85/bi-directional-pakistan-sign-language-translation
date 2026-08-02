"""
Evaluation utilities: load model & dataset, compute confusion matrix, classification report.
Saves `confusion_matrix.png` in the logs folder passed as argument.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow import keras


def evaluate_model(model_path, dataset_path, output_dir='logs/evaluation'):
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(dataset_path)
    # prefer named splits
    if 'X_test' in data:
        X_test = data['X_test']
        y_test = data['y_test']
    else:
        X_test = data['X']
        y_test = data['y']

    with open(os.path.join(os.path.dirname(dataset_path), 'classes.json')) as f:
        classes = json.load(f)

    model = keras.models.load_model(model_path)
    preds = model.predict(X_test)
    y_pred = preds.argmax(axis=1)
    y_true = y_test if y_test.ndim == 1 else y_test.argmax(axis=1)

    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()
    print('Evaluation saved to', output_dir)


if __name__ == '__main__':
    evaluate_model('models/sign_language_model_final.h5', 'processed_data/dataset.npz')
