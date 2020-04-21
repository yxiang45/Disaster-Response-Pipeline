# Disaster Response Pipeline Project

### Description
The motivation of this project is to apply data science and machine learning tools to analyze disaster data from Figure Eight,
and to build a supervised ML model for an API that classifies disaster messages.

### Content
1. ETL Pipeline Preparation.ipynb: loads csv files, clean data and save data to the database.
2. ML Pipeline Preparation.ipynb: loads data, build ML pipeline, train the model and save the model as a pickle file. 
3. models: includes train_classifier.py, a Python script to load data from database, build ML pipeline, train the model and save the model as a pickle file.
4. data: includes two csv data files and process_data.py, a Python script to load csv files, clean data and save data to the database.
5. app: a Flask web app to load a pre-trained ML model and to classify disaster messages.

### Installation
1. Clone or download this repository.
2. Install python3 and Jupyter Notebook.
3. Install required python packages.

### Instructions for execution:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your Disaster Response Pipeline app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001

4. Result:

![alt text](/.images/overview.png "Over view") 
![alt text](/.images/overview1.png "Over view") 
![alt text](/.images/overview2.png "Over view") 
![alt text](/.images/classification1.png "Message classification") 
![alt text](/.images/classification.png "Message classification") 

### Acknowledgements
I would like to thank Udacity for the project design and Figure Eight for Disaster Response dataset.