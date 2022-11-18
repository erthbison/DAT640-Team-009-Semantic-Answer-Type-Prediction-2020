# Semantic Answer Type Prediction

## How to run

Required packages can be found in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Please note that, in case of a CUDA error during the execution, you may need to install pytorch and torchvision the following way:

```bash
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
```

To prepare for Part 1 run the file [`src/round1_prep.py](./src/round1_prep.py). This will preprocess the questions and train the ML predictor.

```bash
python src/round1_prep.py
```

### Baseline Method

To create the entity representation the following files should be extracted from the dbpedia dump and be placed in the [`Data/Dbpedia/`](./Data/dbpedia/) folder.

- `dbpedia_2016.nt`
- `instance_types_en.ttl`
- `long_abstracts_en.ttl`
- `redirects_en.ttl`
- `disambiguations_en.ttl`
- `article_categories_en.ttl`

To preprocess the files and prepare the index the [`src/DBpedia.py`](./src/DBpedia.py) file should be run. It can be run using the following command.

```bash
python src/DBpedia.py
```

After the data has been preprocessed the baseline can be run using the command:

```bash
python src/predictor.py
```

The completed predictions can be found in the following file:  [`Data/baseline_predictions.json`](./Data/baseline_predictions.json)

The scoring can be obtained by running the following command:

```bash
python Data\smart-dataset-master\evaluation\dbpedia\evaluate.py --type_hierarchy_tsv=Data\smart-dataset-master\evaluation\dbpedia\dbpedia_types.tsv --ground_truth_json=Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_test.json --system_output_json=Data/baseline_predictions.json
```

### Advanced Method

````````
For Category Prediction:
  The ML predictor is responsible for identifying the categories as well as the types for the Literal and the boolean category. 
    We therefore use the types as the target for these categories. 
    For questions where the category is resource Part 2 will be used to identify the type, so the category will be used as label.
we run this file 
round1_prep.py
file for category prediction. It will print the accuracy of all three models (SVM, Simple Transformer, Naive Bayes)on console.
``````````````````
For Type Prediction:
predictor.py
``````````````````
 
round1_prep.py 
''''''''''''''''''''''''''''''
BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.
 As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, 
such as question answering and language inference, without substantial task-specific architecture modifications
bert_predictor.py
''''''''''''''''''''''''''''''
For Calculated The accuracy
bert_resource.py

bert_test.py

To evaluate the model run evaluate.py with generated JSON file
