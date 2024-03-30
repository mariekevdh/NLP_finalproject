from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
import numpy as np
import torch
import argparse
import os


def create_arg_parser():
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
        default="da_avg",
        type=str,
        help="Type of qe score used for filtering. Can be 'da_lowest', 'da_avg', 'mqm_lowest', 'mqm_avg' or 'mix' (default 'da_avg').",
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
    score_method: str = "da_avg",
    qe_mix_da_weight: float = 0.5,
    language: str = "nl",
) -> DatasetDict:
    """
    This function returns a DatasetDict with the columns 'premise',
    'hypothesis' and 'label' in the following Datasets:
    train: train data from the GroNLP/ik-nlp-22_transqe dataset
    validation: validation and test data form the GroNLP/ik-nlp-22_transqe dataset
    test: train, validation and test data from the maximedb/sick_nl dataset.

    If the language is 'nl', a column 'qe' will be added to the train Dataset, consisting of either
    the da-score, mqm-score or a mix both scores from the GroNLP/ik-nlp-22_transqe dataset.
    For the da- and mqm-score either the average is taken for the scores of the premise and hypothesis,
    or the lowest of the two values is chosen.

    The data can be filtered on the da-score, mqm-score or a mix both scores
    from the GroNLP/ik-nlp-22_transqe dataset. Can return either Dutch or English data.

    Arguments:
    qe_threshold (float, optional): Quality Estimation score threshold. A float between 0 and 1 (default 0.0).
    score_method (str, optional): Type of qe score used for filtering. Can be 'da_lowest', 'da_avg', 'mqm_lowest', 'mqm_avg' or 'mix' (default 'da_avg').
    qe_mix_da_weight (float, optional): Weight of 'da' score in mixed qe score. A float between 0 and 1 (default 0.5).
    language (str, optional): Language of training data. Can be 'nl' or 'en' (default 'nl').

    Returns: DatasetDict with three Datasets: train, validation and test.
    """

    def average_qe(example):
        premise = float(example["{}_premise".format(score_type)])
        hypothesis = float(example["{}_hypothesis".format(score_type)])
        example["qe"] = (premise + hypothesis) / 2.0
        return example

    def lowest_qe(example):
        premise = float(example["{}_premise".format(score_type)])
        hypothesis = float(example["{}_hypothesis".format(score_type)])
        example["qe"] = min(premise, hypothesis)
        return example

    if score_method.startswith("da"):
        score_type = "da"
    elif score_method.startswith("mqm"):
        score_type = "mqm"

    # Load in base dataset
    dataset = load_dataset("GroNLP/ik-nlp-22_transqe")

    if language == "nl":
        if score_method.endswith("avg"):
            dataset["train"] = dataset["train"].map(average_qe)
        elif score_method.endswith("lowest"):
            dataset["train"] = dataset["train"].map(lowest_qe)
        elif score_method == "mix":
            # qe column = mix
            pass

    if qe_threshold > 0.0:
        filtered_train_data = dataset["train"].filter(
            lambda example: float(example["{}_premise".format(score_type)])
            >= qe_threshold
            and float(example["{}_hypothesis".format(score_type)]) >= qe_threshold
        )
        dataset = DatasetDict(
            {
                "train": filtered_train_data,
                "validation": dataset["validation"],
                "test": dataset["test"],
            }
        )

    # # Load in SICK-NL dataset
    # dataset_sicknl = load_dataset("maximedb/sick_nl")

    # # Rename columns to match base dataset
    # dataset_sicknl = dataset_sicknl.rename_columns(
    #     {
    #         "sentence_A": "premise_nl",
    #         "sentence_B": "hypothesis_nl",
    #         "sentence_A_original": "premise_en",
    #         "sentence_B_original": "hypothesis_en",
    #     }
    # )

    # # Concatenate SICK train, validation and test set and add it as test data in the base dataset
    # dataset["test"] = concatenate_datasets(
    #     [dataset_sicknl["train"], dataset_sicknl["validation"], dataset_sicknl["test"]]
    # ).select_columns(
    #     ["premise_{}".format(language), "hypothesis_{}".format(language), "label"]
    # )

    final_column_renames = {
        "premise_{}".format(language): "premise",
        "hypothesis_{}".format(language): "hypothesis",
    }

    dataset = dataset.rename_columns(final_column_renames)

    select_column_names = ["premise", "hypothesis", "label"]
    if language == "nl":
        select_column_names.append("qe")

    print(dataset)

    return DatasetDict(
        {
            "train": dataset["train"].select_columns(select_column_names),
            "validation": dataset["validation"].select_columns(
                ["premise", "hypothesis", "label"]
            ),
            "test": dataset["test"],
        }
    )


def tokenize_data(dataset, model_name="GroNLP/bert-base-dutch-cased"):
    # Load in tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        return tokenizer(
            example["premise"],
            example["hypothesis"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )

    # Tokenize premise and hypothesis fields
    return dataset.map(tokenize_function, batched=True)


def train_model(
    tokenized_dataset,
    model_name="GroNLP/bert-base-dutch-cased",
    nr_epochs=5,
    weighted_loss=False,
    batch_size=32,
):
    metric = evaluate.load("accuracy", module_type="metric")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Load and finetune model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=nr_epochs,
        evaluation_strategy="epoch",
        logging_dir="logs",
        output_dir="trainer",
        save_total_limit=1,
        fp16=True,
        dataloader_drop_last=True,
        gradient_accumulation_steps=4,
    )

    # Initialize trainer
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

    if args.language == "nl":
        model_name = "GroNLP/bert-base-dutch-cased"
    elif args.language == "en":
        model_name = "google-bert/bert-base-cased"

    dataset = create_dataset(
        qe_threshold=args.qe_threshold,
        score_method=args.score_method,
        qe_mix_da_weight=args.qe_mix_da_weight,
        language=args.language,
    )

    tokenized_dataset = tokenize_data(dataset, model_name=model_name)

    trainer = train_model(
        tokenized_dataset,
        model_name=model_name,
        nr_epochs=args.nr_epochs,
        batch_size=args.batch_size,
    )

    if args.save_model:
        save_path = "bert_{language}_th{qe_threshold}_{score_method}_e{nr_epochs}_b{batch_size}".format(
            language=args.language,
            qe_threshold=args.qe_threshold,
            score_method=args.score_method,
            nr_epochs=args.nr_epochs,
            batch_size=args.batch_size,
        )
        if args.weighted_loss:
            save_path += "_wl"
        if args.save_folder:
            save_path = os.path.join(args.save_folder, save_path)
        trainer.save_model(save_path)

    # # Get predictions on test set and save them to a csv file
    # predictions = trainer.predict(tokenized_dataset["test"])
    # predicted_labels = predictions.predictions.argmax(axis=1)
    # test_dataset_with_predictions = dataset["test"].add_column(
    #     "predicted_label", predicted_labels
    # )
    # test_dataset_with_predictions.to_csv("predictions_nl_full_5epochs.csv", index=False)
