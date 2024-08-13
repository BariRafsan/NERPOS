import config
from sklearn.metrics import classification_report
from source.utils import load_latest_model


def model_eval(X_test, y_pos_test, y_ner_test, pos_encoder, ner_encoder):

    results = load_latest_model(config.model_dir).evaluate(
        X_test, [y_pos_test, y_ner_test]
    )

   
    pos_predictions, ner_predictions = load_latest_model(config.model_dir).predict(
        X_test
    )

  
    pos_predictions = pos_predictions.argmax(axis=-1)
    ner_predictions = ner_predictions.argmax(axis=-1)

    y_pos_test_flat = y_pos_test.flatten()
    y_ner_test_flat = y_ner_test.flatten()
    pos_predictions_flat = pos_predictions.flatten()
    ner_predictions_flat = ner_predictions.flatten()

   
    valid_indices = y_pos_test_flat > 0
    y_pos_test_flat = y_pos_test_flat[valid_indices]
    pos_predictions_flat = pos_predictions_flat[valid_indices]

    valid_indices = y_ner_test_flat > 0
    y_ner_test_flat = y_ner_test_flat[valid_indices]
    ner_predictions_flat = ner_predictions_flat[valid_indices]

   
    pos_report = classification_report(
        y_pos_test_flat, pos_predictions_flat, target_names=pos_encoder.classes_
    )
    ner_report = classification_report(
        y_ner_test_flat, ner_predictions_flat, target_names=ner_encoder.classes_
    )

    return results, pos_report, ner_report
