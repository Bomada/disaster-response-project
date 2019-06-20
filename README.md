# Disaster Response Pipeline Project
Analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set contain real messages that were sent during disaster events. Code create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency. Project also include a web app where an emergency worker can input a new message and get classification results in several categories. The web app also display visualizations of the training data.

### Table of Contents

1. [Instructions](#instruction)
2. [File Descriptions](#file)
3. [Licensing, Authors, Acknowledgements](#license)

### Instructions <a name="instruction"></a>
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database:

    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves it into a .pkl file

    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app:

    `python run.py`

3. Go to http://0.0.0.0:3001/

### File Descriptions <a name="file"></a>
- **app**
    - **templates/go.html**: Web page that handles user query and displays model results
    - **templates/master.html**: Web page with plotly graphs
    - **run.py**: Run web application
- **data**
    - **disaster_categories.csv**: Data for categories which are the target variables
    - **disaster_messages.csv**: Data for messages which features are built from
    - **process_data.py**: Run ETL pipeline that cleans data and stores in database
- **models**
    - **train_classifier.py**: Run ML pipeline that trains classifier and save model into .pkl file

### Licensing, Authors, Acknowledgements <a name="license"></a>
Big thank you to Udacity for providing the template code for this project. Also want to thank Figure Eight for providing the data.
