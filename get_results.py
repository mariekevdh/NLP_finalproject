import argparse
import os

import pandas as pd
from sklearn.metrics import classification_report
from datasets import Dataset, load_dataset


def create_arg_parser() -> argparse.Namespace:
    """
    Creates and returns argument parser for command-line arguments.

    Returns:
        Namespace containing the command-line arguments as attributes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pf",
        "--predictions_folder",
        type=str,
        help="Folder where predictions are stored",
        required=True,
    )
    parser.add_argument(
        "-out",
        "--outfile",
        type=str,
        help="File to store results in. Should be a .csv file.",
        default="results.csv",
    )

    args = parser.parse_args()
    return args


def create_train_dataset(
    qe_threshold: float = 0.0,
    score_method: str = "da",
    qe_mix_da_weight: float = 0.5,
    language: str = "nl",
    baseline: bool = False,
) -> Dataset:
    """
    Note: this is a simplified version of the create_dataset function in finetune_model.py

    Loads and processes the GroNLP/ik-nlp-22_transqe or maximedb/sick_nl dataset from Huggingface.
    The dataset can be filtered on the  parameters specified below.

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
        baseline (bool, optional): If True, loads a baseline dataset (SICK).

    Returns:
        Dataset: The processed dataset filtered using the specified parameters.
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

    da_weight = float(qe_mix_da_weight)

    def mix_scores(example: dict) -> dict:
        """
        Calculates a mixed score for both premise and hypothesis by combining da and mqm scores
        based on a specified weight for the da score in each example. Adds the mixed scores as
        'mix_premise' and 'mix_hypothesis' to the example.
        """
        example["mix_premise"] = float(example["da_premise"]) * da_weight + float(
            example["mqm_premise"]
        ) * (1 - da_weight)
        example["mix_hypothesis"] = float(example["da_hypothesis"]) * da_weight + float(
            example["mqm_hypothesis"]
        ) * (1 - da_weight)
        return example

    def swap_values(example: dict) -> dict:
        """
        Swaps the labels 0 and 2 in a given example. This is needed for the SICK-(NL) dataset
        to match the TransQE dataset.
        """
        if example["label"] == 0:
            example["label"] = 2
        elif example["label"] == 2:
            example["label"] = 0
        return example

    if baseline:
        # Load in and return the train part of the SICK-nl dataset
        dataset_sicknl = load_dataset("maximedb/sick_nl")
        dataset_sicknl = dataset_sicknl.map(swap_values)
        return dataset_sicknl["train"]

    # Load in TransQe dataset
    dataset = load_dataset("GroNLP/ik-nlp-22_transqe")
    dataset = dataset["train"]

    if language == "nl":
        # Scale mqm score to a 0-1 scale
        dataset = dataset.map(scale_mqm)
        if score_type == "mix":
            dataset = dataset.map(mix_scores)
        # Get the lowest score between premise and hypothesis
        dataset = dataset.map(lowest_qe)

    # Filter the data on the given threshold
    if qe_threshold > 0.0:
        dataset = dataset.filter(lambda example: float(example["qe"]) >= qe_threshold)

    return dataset


def get_results(predictions_folder: str) -> pd.DataFrame:
    """
    Processes prediction files from a specified folder. Calculates precision, recall and f1-scores
    per label (Entailment, Neutral and Contradiction), the averages of these scores and counts the
    number of examples that the model was trained on (total and per label).

    Parameters:
        predictions_folder (str): Path to a folder containing prediction csv files.
        They should have a column 'label' (true labels) and 'predictions'

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated results in the following columns:
            - "model_full_name": full name of the model so that it can be linked to prediction files again.
            - "score_method": The score method used ('da', 'mqm' or 'mix'). If 'mix', the da-weight is added as well.
            - "da_weight": Weight of DA score if score method is 'mix'.
            - "threshold": The quality estimation threshold used to filter examples.
            - "weighted_loss": Whether weighted loss was used (True or False).
            - "E_precision": Precision for the Entailment label.
            - "E_recall": Recall for the Entailment label.
            - "E_f1": F1-score for the Entailment label.
            - "N_precision": Precision for the Neutral label.
            - "N_recall": Recall for the Neutral label.
            - "N_f1": F1-score for the Neutral label.
            - "C_precision": Precision for the Contradiction label.
            - "C_recall": Recall for the Contradiction label.
            - "C_f1": F1-score for the Contradiction label.
            - "accuracy": Overall accuracy of the predictions.
            - "weighted_avg_precision": Weighted average precision across all labels.
            - "weighted_avg_recall": Weighted average recall across all labels.
            - "weighted_avg_f1": Weighted average F1-score across all labels.
            - "E_train_ex": Number of training examples for the Entailment label.
            - "N_train_ex": Number of training examples for the Neutral label.
            - "C_train_ex": Number of training examples for the Contradiction label.
            - "total_train_ex": Total number of training examples used.
            - "test_data": Dataset used for predictions: SICK(-NL) or TransQE.
            - "train_data": Dataset used for training: SICK(-NL) or TransQE.
    """
    df_results = pd.DataFrame()

    for file_name in os.listdir(predictions_folder):
        # Get settings from filename
        file_name_parts = file_name[:-4].split("_")
        language = file_name_parts[1]
        baseline = "baseline" in file_name_parts
        weighted_loss = "wl" in file_name_parts
        test_data = f"SICK ({language})"
        if "transqe" in file_name_parts:
            test_data = f"TransQE ({language})"
        model_table = language
        if weighted_loss:
            model_table += " (WL)"
        (
            qe_threshold,
            score_method,
            qe_mix_da_weight,
            qe_mix_da_weight_table,
        ) = (
            0.0,
            "da",
            0.5,
            "-",
        )
        if not baseline:
            qe_threshold = float(file_name_parts[4][2:])
            train_data = f"TransQE ({language})"
            if "mix" in file_name_parts:
                qe_mix_da_weight = file_name_parts[6][8:]
                qe_mix_da_weight_table = qe_mix_da_weight
            score_method = file_name_parts[5]
        else:
            model_table = language + " baseline"
            train_data = f"SICK ({language})"

        df = pd.read_csv(os.path.join(predictions_folder, file_name))

        # Create classification report
        cr = classification_report(
            y_true=df["label"],
            y_pred=df["prediction"],
            labels=[0, 1, 2],
            target_names=["Entailment", "Neutral", "Contradiction"],
            output_dict=True,
        )

        # Recreate dataset to get statistics
        df_dataset = create_train_dataset(
            qe_threshold=float(qe_threshold),
            score_method=score_method,
            qe_mix_da_weight=float(qe_mix_da_weight),
            language=language,
            baseline=baseline,
        ).to_pandas()

        # Put selected results in a new dataframe
        df_new_entry = pd.DataFrame(
            [
                {
                    "model_full_name": file_name[:-4],
                    "score_method": score_method,
                    "da_weight": qe_mix_da_weight_table,
                    "threshold": qe_threshold,
                    "weighted_loss": weighted_loss,
                    "E_precision": cr["Entailment"]["precision"],
                    "E_recall": cr["Entailment"]["recall"],
                    "E_f1": cr["Entailment"]["f1-score"],
                    "N_precision": cr["Neutral"]["precision"],
                    "N_recall": cr["Neutral"]["recall"],
                    "N_f1": cr["Neutral"]["f1-score"],
                    "C_precision": cr["Contradiction"]["precision"],
                    "C_recall": cr["Contradiction"]["recall"],
                    "C_f1": cr["Contradiction"]["f1-score"],
                    "accuracy": cr["accuracy"],
                    "weighted_avg_precision": cr["weighted avg"]["precision"],
                    "weighted_avg_recall": cr["weighted avg"]["recall"],
                    "weighted_avg_f1": cr["weighted avg"]["f1-score"],
                    "E_train_ex": df_dataset["label"].value_counts()[0],
                    "N_train_ex": df_dataset["label"].value_counts()[1],
                    "C_train_ex": df_dataset["label"].value_counts()[2],
                    "total_train_ex": len(df_dataset),
                    "test_data": test_data,
                    "train_data": train_data,
                }
            ]
        )

        print(f"Results added from: {file_name}")

        # Add new dataframe with results from one model to overall results dataframe
        df_results = pd.concat([df_results, df_new_entry], ignore_index=True)

    # Return final dataframe with results for all models
    return df_results


if __name__ == "__main__":
    args = create_arg_parser()

    df_results = get_results(args.predictions_folder)

    # Save results sorted by weighted avg f1 score to csv file
    df_results.round(3).sort_values("weighted_avg_f1", ascending=False).reset_index(
        drop=True
    ).to_csv(args.outfile, index=False)
    print(f"Results saved in {args.outfile}")
