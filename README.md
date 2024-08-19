# Disaster_Recovery_App
Disaster Recovery NLP classification model and web app deployment

The purpose of this project is to build an ETL pipeline to merge, clean and store messages data in a SQL table, 
retrieve the data from the table and pass it through a ML Pipeline which creates a random forest classification model, 
and deploy that model to a web app for the purposes of being an aid in a natural disaster situation. The classification model
can alert relevant parties when a message shows someone is in urgent need of help.

### Instructions on using the .py files and running the app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Files Explanation
- `process_data.py`: Reads text data from csv, performs some data cleaning and processing, and saves to a SQL database. 
- `train_classifier.py`: Reads text model training data from SQL, performs preprocessing required for NLP model training, trains a Random Forest model incorporating
  cross validation for hyperparameter tuning, and saves the model to a pickle file
- `run.py`: Loads the saved Random Forest classifier model. Deploys the trained model to a very basic Flask App which can process text input and serve predictions. 

References:
For this project I used: Udacity course material, reference code from the lessons, and Udacity GPT helper
