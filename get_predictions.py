from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
from datasets import load_dataset, concatenate_datasets, DatasetDict
import argparse
import os
import torch


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mf",
        "--model_folder",
        type=str,
        help="Folder where models are stored",
        default="models/",
    )
    parser.add_argument(
        "-out",
        "--output_folder",
        type=str,
        help="Folder to save the prediction files in. Should be an existing folder.",
        default="predictions/",
    )

    args = parser.parse_args()
    return args


def create_dataset():
    dataset_sicknl = load_dataset("maximedb/sick_nl")
    dataset_sicknl = dataset_sicknl.rename_columns(
        {
            "sentence_A": "premise",
            "sentence_B": "hypothesis",
            "sentence_A_original": "premise_en",
            "sentence_B_original": "hypothesis_en",
        }
    )

    concatenated_dataset = concatenate_datasets(
        [
            dataset_sicknl["train"],
            dataset_sicknl["validation"],
            dataset_sicknl["test"],
        ]
    )

    return DatasetDict(
        {
            "nl": concatenated_dataset.select_columns(
                ["premise", "hypothesis", "label"]
            ),
            "en": concatenated_dataset.select_columns(
                ["premise_en", "hypothesis_en", "label"]
            ).rename_columns({"premise_en": "premise", "hypothesis_en": "hypothesis"}),
        }
    )


def create_input_with_sep(example):
    concatenated_text = f"{example['premise']} [SEP] {example['hypothesis']}"
    return {"concatenated_input": concatenated_text}


def decode_labels(predictions, label2id):
    decoded_predictions = [
        {"prediction": label2id[prediction["label"]], "confidence": prediction["score"]}
        for prediction in predictions
    ]
    return decoded_predictions


def get_predictions(dataset, model_name, tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        max_length=512,
        padding=True,
        truncation=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model, tokenizer=tokenizer, device=device)
    results = classifier(dataset["concatenated_input"])

    label2id = model.config.label2id

    decoded_results = decode_labels(results, label2id)

    predictions = [item['prediction'] for item in decoded_results]
    confidence_scores = [item['confidence'] for item in decoded_results]

    return predictions, confidence_scores


if __name__ == "__main__":
    args = create_arg_parser()
    models_folder = args.model_folder
    output_folder = args.output_folder

    test_data = create_dataset().map(create_input_with_sep)

    if not os.path.isdir(models_folder):
        print(f"The path {models_folder} does not exist or is not a directory.")
    else:
        for model_name in os.listdir(models_folder):
            full_path = os.path.join(models_folder, model_name)
            output_path = os.path.join(output_folder, model_name+".csv")
            output_data = None
            if model_name.startswith("bert_nl"):
                predictions, confidence_scores = get_predictions(
                    test_data["nl"], full_path, "GroNLP/bert-base-dutch-cased"
                )
                output_data = test_data["nl"].add_column("predictions", predictions)
                output_data = output_data.add_column("confidence", confidence_scores)
                output_data.to_csv(output_path)
                print("Predictions saved as {}".format(output_path))
            elif model_name.startswith("bert_en"):
                predictions, confidence_scores = get_predictions(
                    test_data["en"], full_path, "google-bert/bert-base-cased"
                )
                output_data = test_data["en"].add_column("predictions", predictions)
                output_data = output_data.add_column("confidence", confidence_scores)
                output_data.to_csv(output_path)
                print("Predictions saved as {}".format(output_path))