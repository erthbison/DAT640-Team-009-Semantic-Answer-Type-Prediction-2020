Semantic Answer Type Prediction
How to run
Required packages can be found in the requirements.txt file.
pip install -r requirements.txt
To prepare for Part 1 run the file `src/round1_prep.py. This will preprocess the questions and train the ML predictor.
python src/round1_prep.py

To create the entity representation the following files should be extracted from the dbpedia dump and be placed in the Data/Dbpedia/ folder.

dbpedia_2016.nt
instance_types_en.ttl
long_abstracts_en.ttl
redirects_en.ttl
disambiguations_en.ttl
article_categories_en.ttl
To preprocess the files and prepare the index the src/DBpedia.py file should be run. It can be run using the following command.
python src/DBpedia.py
After the data has been preprocessed the baseline can be run using the command:
python src/predictor.py

The completed predictions can be found in the following file: Data/baseline_predictions.json

The scoring can be obtained by running the following command:
python Data\smart-dataset-master\evaluation\dbpedia\evaluate.py --type_hierarchy_tsv=Data\smart-dataset-master\evaluation\dbpedia\dbpedia_types.tsv --ground_truth_json=Data\smart-dataset-master\datasets\DBpedia\smarttask_dbpedia_test.json --system_output_json=Data/baseline_predictions.json

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
bert_predictor.py
bert_resource.py
bert_test.py

To evaluate the model run evaluate.py with generated JSON file
