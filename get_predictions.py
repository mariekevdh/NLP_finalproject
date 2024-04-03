import argparse
import os

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def create_arg_parser() -> argparse.Namespace:
    """
    Creates and returns argument parser for command-line arguments.

    Returns:
        Namespace containing the command-line arguments as attributes.
    """
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
        help="Folder to save the prediction files in.",
        default="predictions/",
    )

    args = parser.parse_args()
    return args


def create_dataset() -> DatasetDict:
    """
    Loads in the maximedb/sick_nl dataset from HuggingFace.

    Returns:
        DatasetDict: DatasetDict containing two Datasets containing the premise,
            hypothesis and labels: 'nl' (Dutch) and 'en' (English)
    """
    # Load in the sick dataset and rename columns to better understandable names
    dataset_sicknl = load_dataset("maximedb/sick_nl")
    dataset_sicknl = dataset_sicknl.rename_columns(
        {
            "sentence_A": "premise",
            "sentence_B": "hypothesis",
            "sentence_A_original": "premise_en",
            "sentence_B_original": "hypothesis_en",
        }
    )

    # Return dataset containing a Dutch and English split.
    return DatasetDict(
        {
            "nl": dataset_sicknl["test"].select_columns(
                ["premise", "hypothesis", "label"]
            ),
            "en": dataset_sicknl["test"]
            .select_columns(["premise_en", "hypothesis_en", "label"])
            .rename_columns({"premise_en": "premise", "hypothesis_en": "hypothesis"}),
        }
    )


def create_input_with_sep(example: dict) -> dict:
    """
    Concatenates the 'premise' and 'hypothesis' fields from a given example with
    a '[SEP]' token for easier use in a Transformers pipeline. Intended to be used
    on a DatasetDict using the map() method.
    """
    concatenated_text = f"{example['premise']} [SEP] {example['hypothesis']}"
    return {"concatenated_input": concatenated_text}


def decode_labels(predictions: list[dict], label2id: dict) -> list[dict]:
    """
    Decodes prediction labels into their original integer representation. Also renames
    the 'label' and 'score' columns to 'prediction' and 'confidence' to avoid mix-ups.

    Parameters:
        predictions (list[dict]): A list of dictionaries, each containing the keys 'label'
            and 'score' (confidence score of the prediction).
        label2id (dict): A dictionary mapping the labels as given by the model
        to the original representations.

    Returns:
        list[dict]: A list of dictionaries, each containing the keys 'prediction' and 'confidence'.
    """
    decoded_predictions = [
        {"prediction": label2id[prediction["label"]], "confidence": prediction["score"]}
        for prediction in predictions
    ]
    return decoded_predictions


def get_predictions(
    dataset: Dataset, model_name: str, tokenizer_name: str
) -> tuple[list, list]:
    """
    Loads in a tokenizer and sequence classification model and gets predictions
    using Hugging Face's text-classification pipeline.

    Parameters:
        dataset (Dataset): A dataset containing the input data under the key 'concatenated_input'.
        model_name (str): The name or path of the pretrained model to load.
        tokenizer_name (str): The name or path of the tokenizer to load.

    Returns:
        tuple[list, list]: Two lists, one containing predictions and one containing confidence scores.
    """
    # Load in tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        max_length=512,
        padding=True,
        truncation=True,
    )

    # Load in model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Create classifier pipeline, using gpu if available and get results
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "text-classification", model, tokenizer=tokenizer, device=device
    )
    results = classifier(dataset["concatenated_input"])

    # Get label mapping from model and decode labels
    label2id = model.config.label2id
    decoded_results = decode_labels(results, label2id)

    # Create two lists of predictions and confidence scores and return them
    predictions = [item["prediction"] for item in decoded_results]
    confidence_scores = [item["confidence"] for item in decoded_results]

    return predictions, confidence_scores


if __name__ == "__main__":
    args = create_arg_parser()
    models_folder = args.model_folder
    output_folder = args.output_folder

    # Create test dataset and concatenate premise and hypothesis using [SEP] as separator
    test_data = create_dataset().map(create_input_with_sep)

    # Check if model and output folders exist
    if not os.path.isdir(models_folder):
        print(f"The path {models_folder} does not exist or is not a directory.")
    else:
        # Loop though models in model folder to get predictions
        for model_name in os.listdir(models_folder):
            # Create full paths using model name
            full_path = os.path.join(models_folder, model_name)
            output_path = os.path.join(output_folder, model_name + ".csv")
            output_data = None
            if model_name.startswith("bert_nl"):
                # If the model is Dutch, use Dutch data to get predictions
                predictions, confidence_scores = get_predictions(
                    test_data["nl"], full_path, "GroNLP/bert-base-dutch-cased"
                )
                output_data = test_data["nl"].add_column("predictions", predictions)
            elif model_name.startswith("bert_en"):
                # If the model is English, use English data to get predictions
                predictions, confidence_scores = get_predictions(
                    test_data["en"], full_path, "google-bert/bert-base-cased"
                )
                output_data = test_data["en"].add_column("predictions", predictions)
            # Add confidence scores to dataset with predictions and remove concatenated_input
            output_data = output_data.add_column("confidence", confidence_scores)
            output_data = output_data.remove_columns("concatenated_input")
            # Check if output folder exists, if not: create it
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
                print(f"Created output folder: {output_folder}")
            # Save final dataset original data and predictions and confidence scores to csv file.
            output_data.to_csv(output_path)
            print("Predictions saved as {}".format(output_path))
