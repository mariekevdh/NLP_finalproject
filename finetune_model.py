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
import torch.nn.functional as F


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
    baseline: bool = False
) -> DatasetDict:
    """
    This function returns a DatasetDict with the columns 'premise',
    'hypothesis' and 'labels' in the following Datasets:
    train: train data from the GroNLP/ik-nlp-22_transqe dataset
    validation: validation and test data form the GroNLP/ik-nlp-22_transqe dataset
    test: train, validation and test data from the maximedb/sick_nl dataset.

    If the language is 'nl', a column 'qe' will be added to the train Dataset, consisting of either
    the da-score, mqm-score or a mix both scores from the GroNLP/ik-nlp-22_transqe dataset.
    For the da- and mqm-score the lowest of the hypothesis and premise values is chosen.

    The data can be filtered on the da-score, mqm-score or a mix both scores
    from the GroNLP/ik-nlp-22_transqe dataset. Can return either Dutch or English data.

    Arguments:
    qe_threshold (float, optional): Quality Estimation score threshold. A float between 0 and 1 (default 0.0).
    score_method (str, optional): Type of qe score used for filtering. Can be 'da', 'mqm', or 'mix' (default 'da').
    qe_mix_da_weight (float, optional): Weight of 'da' score in mixed qe score. A float between 0 and 1 (default 0.5).
    language (str, optional): Language of training data. Can be 'nl' or 'en' (default 'nl').

    Returns: DatasetDict with three Datasets: train, validation and test.
    """

    score_type = score_method
    def lowest_qe(example):
        premise = float(example["{}_premise".format(score_type)])
        hypothesis = float(example["{}_hypothesis".format(score_type)])
        example["qe"] = min(premise, hypothesis)
        return example

    def scale_mqm(example):
        example["mqm_premise"] = float(example["mqm_premise"]) / 0.2
        example["mqm_hypothesis"] = float(example["mqm_hypothesis"]) / 0.2
        return example

    qe_mix_da_weight = float(qe_mix_da_weight)

    def mix_scores(example):
        example["mix_premise"] = float(
            example["da_premise"]
        ) * qe_mix_da_weight + float(example["mqm_premise"]) * (1 - qe_mix_da_weight)
        example["mix_hypothesis"] = float(
            example["da_hypothesis"]
        ) * qe_mix_da_weight + float(example["mqm_hypothesis"]) * (1 - qe_mix_da_weight)
        return example

    def swap_values(example):
        if example['label'] == 0:
            example['label'] = 2
        elif example['label'] == 2:
            example['label'] = 0
        return example

    # Load in base dataset
    dataset = load_dataset("GroNLP/ik-nlp-22_transqe")
    dataset_sicknl = load_dataset("maximedb/sick_nl")

    dataset_sicknl = dataset_sicknl.rename_columns(
        {
            "sentence_A": "premise_nl",
            "sentence_B": "hypothesis_nl",
            "sentence_A_original": "premise_en",
            "sentence_B_original": "hypothesis_en",
        }
    )

    dataset_sicknl = dataset_sicknl.map(swap_values)

    if language == "nl":
        # Scale mqm score to a 0-1 scale
        dataset["train"] = dataset["train"].map(scale_mqm)
        if score_type == "mix":
            dataset["train"] = dataset["train"].map(mix_scores)
        dataset["train"] = dataset["train"].map(lowest_qe)
    if baseline:
        dataset["train"] = dataset_sicknl["train"]
    dataset["validation"] = dataset_sicknl["validation"]

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

    final_column_renames = {
        "premise_{}".format(language): "premise",
        "hypothesis_{}".format(language): "hypothesis",
        "label": "labels",
    }

    dataset = dataset.rename_columns(final_column_renames)

    select_column_names = ["premise", "hypothesis", "labels"]
    if language == "nl" and not baseline:
        select_column_names.append("qe")

    final_dataset = DatasetDict(
        {
            "train": dataset["train"].select_columns(select_column_names),
            "validation": dataset["validation"].select_columns(["premise", "hypothesis", "labels"]),
        }
    )

    print(final_dataset)
    return final_dataset


def tokenize_data(dataset, tokenizer):
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


class TrainerWithQeWeights(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
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
        save_strategy='no',
        fp16=True,
        dataloader_drop_last=True,
        gradient_accumulation_steps=4,
        remove_unused_columns=remove_unused_columns,
    )

    # Initialize trainer
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
    print('data check')
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
        baseline=args.baseline
    )

    if not args.check_data:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, max_length=512, padding=True, truncation=True
        )
        tokenized_dataset = tokenize_data(dataset, tokenizer)

        trainer = train_model(
            tokenized_dataset,
            model_name=model_name,
            nr_epochs=args.nr_epochs,
            batch_size=args.batch_size,
            weighted_loss=args.weighted_loss,
        )

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
