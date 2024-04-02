import argparse
import os
from typing import Union

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


def create_arg_parser() -> argparse.Namespace:
    """
    Creates and returns argument parser for command-line arguments.

    Returns:
        Namespace containing the command-line arguments as attributes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-th",
        "--qe_threshold",
        default=0.0,
        type=float,
        help="Quality Estimation score threshold. A float between 0 and 1 (default 0.0).",
    )
    parser.add_argument(
        "-qe",
        "--score_method",
        default="da",
        type=str,
        help="Type of qe score used for filtering. Can be 'da', 'mqm', or 'mix' (default 'da').",
    )
    parser.add_argument(
        "-w",
        "--qe_mix_da_weight",
        default=0.5,
        type=float,
        help="Weight of 'da' score in mixed qe score. A float between 0 and 1 (default 0.5).",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="nl",
        type=str,
        help="Language of training data. Can be 'nl' or 'en' (default 'nl').",
    )
    parser.add_argument(
        "-e",
        "--nr_epochs",
        default=4,
        type=int,
        help="Number of epochs to fine-tune the model for.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for fine-tuning the model.",
    )
    parser.add_argument(
        "-wl",
        "--weighted_loss",
        default=False,
        type=bool,
        help="Indicates whether or not a weighted loss function is used for fine-tuning.",
    )
    parser.add_argument(
        "-base",
        "--baseline",
        default=False,
        type=bool,
        help="If True, the sick dataset will be used to train.",
    )
    parser.add_argument(
        "-data",
        "--check_data",
        default=False,
        type=bool,
        help="If True, the script will create the dataset with the given settings and print out the structure and numbers.",
    )
    parser.add_argument(
        "-sm",
        "--save_model",
        default=True,
        type=bool,
        help="Indicates whether or not to save the fine-tuned model",
    )
    parser.add_argument(
        "-sf", "--save_folder", type=str, help="Folder to save the model in"
    )

    args = parser.parse_args()
    return args


def create_dataset(
    qe_threshold: float = 0.0,
    score_method: str = "da",
    qe_mix_da_weight: float = 0.5,
    language: str = "nl",
    baseline: bool = False,
) -> DatasetDict:
    """
    Loads and processes the GroNLP/ik-nlp-22_transqe and maximedb/sick_nl dataset from Huggingface.
    The GroNLP/ik-nlp-22_transqe dataset can be filtered on the parameters specified below.
    The evalutaion part of the maximedb/sick_nl will be added as evalutation data.
    If the parameter baseline is True, the train part of this dataset will be added as train data.

    Parameters:
        qe_threshold (float, optional): The minimum quality estimation score
            for including an example in the dataset. The lowest score between
            the premise- and hypothesis-score will be used for filtering.
            Defaults to 0.0, which includes all examples.
        score_method (str, optional): The scoring method to use. Can be 'da' (Direct Assessment),
            'mqm' (Multidimensional Quality Metrics), or 'mix' for a weighted mix of 'da' and 'mqm' scores.
            Defaults to 'da'. 'mqm' will be scaled to a 0-1 scale to match 'da'.
        qe_mix_da_weight (float, optional): The weight for the 'da' score in the mixed score calculation.
            Only relevant if `score_method` is set to 'mix'. Defaults to 0.5.
        language (str, optional): The language of the dataset to load. Filtering is only possible for 'nl' (Dutch).
            Defaults to 'nl'.
        baseline (bool, optional): If True, adds a baseline dataset (SICK) as train data.

    Returns:
        DatasetDict: The processed dataset filtered using the specified parameters.
        Contains a Dataset called 'train' and a Dataset called 'validation'.
    """

    score_type = score_method

    def lowest_qe(example: dict) -> dict:
        """
        Calculates the minimum QE score between premise and hypothesis
        scores in the given example and adds it to the example under the 'qe' key.
        """
        premise = float(example["{}_premise".format(score_type)])
        hypothesis = float(example["{}_hypothesis".format(score_type)])
        example["qe"] = min(premise, hypothesis)
        return example

    def scale_mqm(example: dict) -> dict:
        """
        Scales the mqm score in an example to a 0-1 scale.
        """
        example["mqm_premise"] = float(example["mqm_premise"]) / 0.2
        example["mqm_hypothesis"] = float(example["mqm_hypothesis"]) / 0.2
        return example

    qe_mix_da_weight = float(qe_mix_da_weight)

    def mix_scores(example: dict) -> dict:
        """
        Calculates a mixed score for both premise and hypothesis by combining da and mqm scores
        based on a specified weight for the da score in each example. Adds the mixed scores as
        'mix_premise' and 'mix_hypothesis' to the example.
        """
        example["mix_premise"] = float(
            example["da_premise"]
        ) * qe_mix_da_weight + float(example["mqm_premise"]) * (1 - qe_mix_da_weight)
        example["mix_hypothesis"] = float(
            example["da_hypothesis"]
        ) * qe_mix_da_weight + float(example["mqm_hypothesis"]) * (1 - qe_mix_da_weight)
        return example

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

    # Load in base datasets
    dataset = load_dataset("GroNLP/ik-nlp-22_transqe")
    dataset_sicknl = load_dataset("maximedb/sick_nl")

    # Rename columns in sick dataset to match the transqe dataset
    dataset_sicknl = dataset_sicknl.rename_columns(
        {
            "sentence_A": "premise_nl",
            "sentence_B": "hypothesis_nl",
            "sentence_A_original": "premise_en",
            "sentence_B_original": "hypothesis_en",
        }
    )

    # Swap the labels 0 and 2 in the sick dataset
    dataset_sicknl = dataset_sicknl.map(swap_values)

    if language == "nl":
        # Scale mqm score to a 0-1 scale
        dataset["train"] = dataset["train"].map(scale_mqm)
        if score_type == "mix":
            # Add mixed qe scores
            dataset["train"] = dataset["train"].map(mix_scores)
        # Get lowest qe scores between premise and hypothesis
        dataset["train"] = dataset["train"].map(lowest_qe)
    if baseline:
        # Add sick data as train data
        dataset["train"] = dataset_sicknl["train"]
    # Add sick data as validation data
    dataset["validation"] = dataset_sicknl["validation"]

    # Filter the dataset on the given threshold
    if qe_threshold > 0.0:
        filtered_train_data = dataset["train"].filter(
            lambda example: float(example["qe"]) >= qe_threshold
        )

        dataset = DatasetDict(
            {
                "train": filtered_train_data,
                "validation": dataset["validation"],
            }
        )

    # Rename premise and hypothesis columns to not include a language,
    # change label to labels.
    final_column_renames = {
        "premise_{}".format(language): "premise",
        "hypothesis_{}".format(language): "hypothesis",
        "label": "labels",
    }

    dataset = dataset.rename_columns(final_column_renames)

    # Keep only the premise, hypothesis, labels and - if present - qe columns in the dataset
    select_column_names = ["premise", "hypothesis", "labels"]
    if language == "nl" and not baseline:
        select_column_names.append("qe")

    final_dataset = DatasetDict(
        {
            "train": dataset["train"].select_columns(select_column_names),
            "validation": dataset["validation"].select_columns(
                ["premise", "hypothesis", "labels"]
            ),
        }
    )

    return final_dataset


def tokenize_data(dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
    """
    Tokenizes the premise and hypothesis fields of the input data using the given tokenizer.
    A [SEP] token will be added between the fields by the tokenizer, since this is expected by
    BERT-based models.

    Parameters:
        dataset (Dataset): Dataset that contains a premise and hypothesis field
        tokenizer (PreTrainedTokenizer): Insance of a pretrained tokenizer

    Returns:
        Dataset with tokenized input
    """

    def tokenize_function(example: dict) -> dict:
        return tokenizer(
            example["premise"],
            example["hypothesis"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    # Tokenize premise and hypothesis fields
    return dataset.map(tokenize_function, batched=True)


class TrainerWithQeWeights(Trainer):
    """
    A custom trainer class that extends HuggingFace's Trainer to support
    weighted loss computation based on quality estimation (QE) weights.

    Methods:
        compute_loss(model, inputs, return_outputs=False):
            Computes the loss using QE weights. If QE weights are missing or invalid,
            it returns the unweighted loss.
    """

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """
        Computes the loss for a batch of inputs using QE scores as weights.

        Parameters:
            model (PreTrainedModel): The model being finetuned.
            inputs (dict[str, torch.Tensor]): Inputs to the model, should have 'labels' and 'qe' column.
            return_outputs (bool, optional): If True, the method also returns the model's outputs.
                Defaults to False.

        Returns:
            Union[torch.Tensor, tuple[torch.Tensor, dict]]: The computed loss tensor or a tuple
                of the loss and the model's outputs if return_outputs is set to True.
        """
        labels = inputs.pop("labels")
        qe_weights = inputs.pop("qe")
        outputs = model(**inputs)
        logits = outputs.logits

        # Convert qe_weights from strings to floats
        qe_weights = torch.tensor([float(q) for q in qe_weights], device=model.device)

        # Compute the loss for each sample
        loss_per_sample = F.cross_entropy(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
            reduction="none",
        )

        # Check if qe weights are there and have the correct shape
        if qe_weights is not None and qe_weights.shape[0] == loss_per_sample.shape[0]:
            # Apply sample weights and calculate average
            weighted_loss = loss_per_sample * qe_weights
            loss = weighted_loss.mean()
        else:
            # if no or missing weights, compute mean loss without weights
            loss = loss_per_sample.mean()

        if return_outputs:
            return (loss, outputs)
        return loss


def train_model(
    tokenized_dataset: DatasetDict,
    model_name: str = "GroNLP/bert-base-dutch-cased",
    nr_epochs: int = 5,
    weighted_loss: bool = False,
    batch_size: int = 32,
) -> Trainer:
    """
    Trains a model on a given tokenized dataset for sequence classification.

    Parameters:
        tokenized_dataset (DatasetDict): A tokenized dataset. Should include 'train' and 'validation' splits.
        model_name (str, optional): Modelname from HuggingFace's model hub. Defaults to "GroNLP/bert-base-dutch-cased".
        nr_epochs (int, optional): Number of epochs to train the model for. Defaults to 5.
        weighted_loss (bool, optional): Indicates whether to use qe-based weighted loss during training.
            If True, requires qe scores in the dataset. Defaults to False.
        batch_size (int, optional): Batch size for training and evaluation. Defaults to 32.

    Returns:
        Trainer: A trained Trainer with the fine-tuned model.
    """
    metric = evaluate.load("accuracy", module_type="metric")

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        """
        Computes the evaluation metrics for the model's predictions.
        metric should be defined within this function's scope.
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Load and finetune model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # If weighted loss is used, the Trainer needs to keep the column containing the qe scores.
    remove_unused_columns = True
    if weighted_loss:
        remove_unused_columns = False

    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=nr_epochs,
        evaluation_strategy="epoch",
        logging_dir="logs",
        output_dir="trainer",
        save_strategy="no",
        fp16=True,
        dataloader_drop_last=True,
        gradient_accumulation_steps=4,
        remove_unused_columns=remove_unused_columns,
    )

    # Initialize trainer. If weighted_loss is set to True, use a custom Trainer
    # that supports weighted loss, otherwise use the standard Trainer.
    if weighted_loss:
        trainer = TrainerWithQeWeights(
            model=model.to(device),
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model.to(device),
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=compute_metrics,
        )

    # Train and return model
    trainer.train()
    return trainer


if __name__ == "__main__":
    args = create_arg_parser()

    # Create the (optionally filtered) dataset
    dataset = create_dataset(
        qe_threshold=args.qe_threshold,
        score_method=args.score_method,
        qe_mix_da_weight=args.qe_mix_da_weight,
        language=args.language,
        baseline=args.baseline,
    )

    print(dataset)

    # If check_data is not set to True, load and train model
    if not args.check_data:
        # Load in the correct model based on the given language
        if args.language == "nl":
            model_name = "GroNLP/bert-base-dutch-cased"
        elif args.language == "en":
            model_name = "google-bert/bert-base-cased"

        # Load in the correct tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, max_length=512, padding=True, truncation=True
        )
        tokenized_dataset = tokenize_data(dataset, tokenizer)

        # Train the model
        trainer = train_model(
            tokenized_dataset,
            model_name=model_name,
            nr_epochs=args.nr_epochs,
            batch_size=args.batch_size,
            weighted_loss=args.weighted_loss,
        )

        # If save_model is set to True, dynamically create a folder name based on
        # the provided parameters to save the model in. If a save_folder is given,
        # the model will be stored there (in a subfolder with the dynamically created folder name)
        if args.save_model:
            save_path = f"bert_{args.language}_e{args.nr_epochs}_b{args.batch_size}"
            if args.baseline:
                save_path += "_baseline"
            else:
                save_path += f"_th{args.qe_threshold}_{args.score_method}"
            if args.score_method == "mix":
                save_path += f"_daweight{args.qe_mix_da_weight}"
            if args.weighted_loss:
                save_path += "_wl"

            if args.save_folder:
                save_path = os.path.join(args.save_folder, save_path)
            trainer.save_model(save_path)
        print("File saved as in folder: {}".format(save_path))
