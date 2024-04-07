# Translated data for fine-tuning: quantity vs. quality

This repository contains the code that was used for the final project of the course Natural Language Processing (LIX001M05) at the University of Groningen. The project explores whether data quantity or quality is more important when fine-tuning a pre-trained language model for an NLI task.

The dataset that we use for fine-tuning is an automatically translated version of the E-SNLI dataset called TransQE. Besides the original premises, hypotheses, labels and explanations, it also contains automatically translated (to Dutch) versions of the text fields, as well as automatically generated quality estimation scores for each text field. The models are evaluated on the SICK-NL dataset, which contains Dutch translations of the SICK dataset that were verified and/or edited by humans.

An overview of our results can be explored using interactive tables in the Jupyter Notebook analysis.ipynb or alternatively on the HTML version of the notebook: https://translated-data-for-finetuning-quantity-vs-quality.tiiny.site/

## Experiments
Our experiments consist of fine-tuning the Dutch BERT-based model BERTje on different subsets of the TransQE dataset filtered based on the different quality estimation scores. As a baseline we use the English BERT model fine-tuned on the original English E-SNLI dataset.

### How to run the experiments
#### Replicate experiments using SLURM
We ran our experiments on the Hábrók high performance computing cluster which uses the SLURM resource scheduler. To replicate our experiments using the SLURM scheduler you can run the following commands:

```bash
# Create environment
bash jobscripts/create_environment.sh
```

```bash
# Run experiments. Make sure environment has been created
bash jobscripts/run_experiments.sh
```

```bash
# Get predictions using SICK data. Make sure all models are done fine-tuning
bash jobscripts/get_predictions.sh
# Get predictions of baseline model on TransQE data
bash jobscripts/get_predictions_transqe.sh
```

```bash
# Get results. Make sure predictions have been created.
bash jobscripts/get_results.sh
```

#### Run experiments without SLURM
Alternatively, you can run the code by calling the scripts directly. To install the dependencies, run the following line in your terminal (depending on your system you might have to use pip3):

```bash
pip install -r requirements.txt
```

To run experiments, run the following line. You can customize the fine-tuning process with the following command line arguments (depending on your system you might have to use python3):

```bash
python finetune_model.py
```

#### Arguments (all optional):

- **`-th` `--qe_threshold`**  
  **Default**: `0.0`  
  **Description**: Quality Estimation score threshold. Specify a float between 0 and 1.

- **`-qe` `--score_method`**  
  **Default**: `"da"`  
  **Description**: Type of QE score used for filtering. Options are `'da'`, `'mqm'`, or `'mix'`.

- **`-w` `--qe_mix_da_weight`**  
  **Default**: `0.5`  
  **Description**: Weight of 'da' score in mixed QE score. Specify a float between 0 and 1.

- **`-l` `--language`**  
  **Default**: `"nl"`  
  **Description**: Language of training data. Can be `'nl'` or `'en'`.

- **`-e` `--nr_epochs`**  
  **Default**: `4`  
  **Description**: Number of epochs to fine-tune the model for.

- **`-b` `--batch_size`**  
  **Default**: `32`  
  **Description**: Batch size for fine-tuning the model.

- **`-wl` `--weighted_loss`**  
  **Default**: `False`  
  **Description**: Indicates whether or not a weighted loss function is used for fine-tuning.

- **`-base` `--baseline`**  
  **Default**: `False`  
  **Description**: If True, uses the SICK dataset to train.

- **`-data` `--check_data`**  
  **Default**: `False`  
  **Description**: If True, creates the dataset with the given settings and prints out its structure and numbers.

- **`-sm` `--save_model`**  
  **Default**: `True`  
  **Description**: Indicates whether or not to save the fine-tuned model.

- **`-sf` `--save_folder`**  
  **Default**: `"models/"`  
  **Description**: Folder to save the model in.

To get predictions from your fine-tuned model(s), you can use the following script:
```bash
python get_predictions.py
```
#### Arguments (all optional):

- **`-mf` `--model_folder`**  
  **Default**: `"models/"`  
  **Description**: Folder where models are stored.

- **`-mn` `--model_name`**  
  **Default**: `None`  
  **Description**: Model name. Specify the name of a specific model if you want predictions on one model only. The model should exist in the folder specified with `--model_folder`.

- **`-out` `--output_folder`**  
  **Default**: `"predictions/"`  
  **Description**: Folder to save the prediction files in.

- **`-d` `--dataset`**  
  **Default**: `"sicknl"`  
  **Description**: Indicates which dataset to load in: `'sicknl'` or `'transqe'`.

- **`-b` `--batch_size`**  
  **Default**: `16`  
  **Description**: Batch size to use for predictions.


After you get the predictions, you can get results using the following script (assuming your predictions are in the folder 'predictions', change according to your folder structure):

```bash
python3 get_results.py -pf 'predictions/'
```
#### Required Arguments

- **`-pf` `--predictions_folder`**  
  **Description**: The folder where predictions are stored. This argument is required and must be specified by the user.

#### Optional Arguments

- **`-out` `--outfile`**  
  **Default**: `"results.csv"`  
  **Description**: The file to store results in. It should be a .csv file.


### Models used
[🤗 BERT](https://huggingface.co/google-bert/bert-base-cased) \
[🤗 BERTje](https://huggingface.co/GroNLP/bert-base-dutch-cased)

### Datasets used
[🤗 TransQE](https://huggingface.co/datasets/GroNLP/ik-nlp-22_transqe) \
[🤗 SICK-NL](https://huggingface.co/datasets/maximedb/sick_nl)