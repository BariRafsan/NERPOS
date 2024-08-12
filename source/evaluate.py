from icecream import ic
from sklearn.metrics import classification_report
from source.utils import (
    X_test,
    y_pos_test,
    y_ner_test,
    pos_encoder,
    ner_encoder,
)
from source.utils import model


# Load and preprocess data


results = model.evaluate(X_test, [y_pos_test, y_ner_test])
ic(results)

# Step 2: Predict on the test set
pos_predictions, ner_predictions = model.predict(X_test)

# Convert predictions to the label format expected by classification_report
# Note: This step is necessary because classification_report expects integers (class labels)
pos_predictions = pos_predictions.argmax(axis=-1)
ner_predictions = ner_predictions.argmax(axis=-1)

y_pos_test_flat = y_pos_test.flatten()
y_ner_test_flat = y_ner_test.flatten()
pos_predictions_flat = pos_predictions.flatten()
ner_predictions_flat = ner_predictions.flatten()

# Remove padding (where y_test values are 0, which is usually the padding index)
valid_indices = y_pos_test_flat > 0
y_pos_test_flat = y_pos_test_flat[valid_indices]
pos_predictions_flat = pos_predictions_flat[valid_indices]

valid_indices = y_ner_test_flat > 0
y_ner_test_flat = y_ner_test_flat[valid_indices]
ner_predictions_flat = ner_predictions_flat[valid_indices]

# Step 3: Calculate precision, recall, and F1 score
pos_report = classification_report(
    y_pos_test_flat, pos_predictions_flat, target_names=pos_encoder.classes_
)
ner_report = classification_report(
    y_ner_test_flat, ner_predictions_flat, target_names=ner_encoder.classes_
)

print("POS Tagging Report:\n", pos_report)
print("NER Tagging Report:\n", ner_report)
