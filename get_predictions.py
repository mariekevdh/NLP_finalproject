import argparse
import os

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Indicates which dataset to load in: 'sicknl' or 'transqe'.",
        default="sicknl",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Batch size to use for predictions.",
        default=16,
    )

    args = parser.parse_args()
    return args


def create_dataset(dataset: str = "sicknl") -> DatasetDict:
    """
    Returns the test split of the maximedb/sick_nl or GroNLP/ik-nlp-22_transqe
    dataset from HuggingFace.

    Parameters:
        dataset (str): The dataset to load in. Can be 'sicknl' or 'transqe'.
            Default is 'sicknl'.

    Returns:
        DatasetDict: DatasetDict containing two Datasets containing the premise,
            hypothesis and labels: "nl" (Dutch) and "en" (English)
    """

    def swap_values(example: dict) -> dict:
        """
        Swaps the labels 0 and 2 in a given example. At the time of writing,
        this is needed for the SICK-NL dataset to match the TransQE dataset.
        """
        if example["label"] == 0:
            example["label"] = 2
        elif example["label"] == 2:
            example["label"] = 0
        return example

    if dataset == "sicknl":
        # Load in the sick dataset and rename columns to better understandable names
        dataset = load_dataset("maximedb/sick_nl")
        dataset = dataset.rename_columns(
            {
                "sentence_A": "premise_nl",
                "sentence_B": "hypothesis_nl",
                "sentence_A_original": "premise_en",
                "sentence_B_original": "hypothesis_en",
            }
        )
        dataset = dataset.map(swap_values)
    elif dataset == "transqe":
        dataset = load_dataset("GroNLP/ik-nlp-22_transqe")
    # Return dataset containing a Dutch and English split.
    return DatasetDict(
        {
            "nl": dataset["test"]
            .select_columns(["premise_nl", "hypothesis_nl", "label"])
            .rename_columns({"premise_nl": "premise", "hypothesis_nl": "hypothesis"}),
            "en": dataset["test"]
            .select_columns(["premise_en", "hypothesis_en", "label"])
            .rename_columns({"premise_en": "premise", "hypothesis_en": "hypothesis"}),
        }
    )


def get_predictions(dataset: Dataset, model_name: str, batch_size: int = 16) -> Dataset:
    """
    Loads in a tokenizer and sequence classification model and gets predictions
    using Hugging Face"s text-classification pipeline.

    Parameters:
        dataset (Dataset): A dataset containing the input data under the key "concatenated_input".
        model_name (str): The name or path of the pretrained model to load.
        batch_size (int): Batch size to use for the tokenization

    Returns:
        Dataset: Dataset with columns 'premise', 'hypothesis', 'label' and 'prediction'.
    """
    # Load in model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    # Load in tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def batch_predict(batch) -> dict:
        """
        Processes a batch of inputs: tokenizes the input and predicts a label.
        """
        # Tokenize the batch
        inputs = tokenizer(
            batch["premise"],
            batch["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        # Put input on same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

        # Return predictions in CPU and convert to list
        return {"prediction": predictions.cpu().numpy().tolist()}

    dataset_preds = dataset.map(batch_predict, batched=True, batch_size=batch_size)
    return dataset_preds.select_columns(
        ["premise", "hypothesis", "label", "prediction"]
    )


if __name__ == "__main__":
    args = create_arg_parser()
    models_folder = args.model_folder
    output_folder = args.output_folder

    # Create test dataset
    test_data = create_dataset(dataset=args.dataset)

    # Check if model and output folders exist
    if not os.path.isdir(models_folder):
        print(f"The path {models_folder} does not exist or is not a directory.")
    else:
        # Loop though models in model folder to get predictions
        for model_name in os.listdir(models_folder):
            # Create full paths using model name
            full_model_path = os.path.join(models_folder, model_name)
            output_path = os.path.join(output_folder, model_name + ".csv")
            if model_name.startswith("bert_nl"):
                # If the model is Dutch, use Dutch data to get predictions
                predictions = get_predictions(
                    test_data["nl"], full_model_path, batch_size=args.batch_size
                )
            elif model_name.startswith("bert_en"):
                # If the model is English, use English data to get predictions
                predictions = get_predictions(
                    test_data["en"], full_model_path, batch_size=args.batch_size
                )

            # Check if output folder exists, if not: create it
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
                print(f"Created output folder: {output_folder}")
            # Save final dataset with predictions to csv file.
            predictions.to_csv(output_path)
            print("Predictions saved to {}".format(output_path))
