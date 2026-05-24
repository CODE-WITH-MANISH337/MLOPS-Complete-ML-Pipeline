import pandas as pd
import os
import logging 
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import json
import pickle
import numpy as np


log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(file_path:str):
    """Load the trained model form file"""
    try:
        with open(file_path,'rb') as f:
            model=pickle.load(f)
        logger.debug('Model load from %s',file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found %s',file_path)
        raise
    except Exception as e:
        logger.error('Unexcepted error occurred while loading the file %s',file_path)
        raise

def load_data(file_path:str)->pd.DataFrame:
    """Loading the test data"""
    try:
        df=pd.read_csv(file_path)
        logger.debug('Data Loaded from %s',file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexcepted error occurred while loading the file %s',file_path)
        raise

def evaluate_model(clf,X_test:np.ndarray, y_test:np.ndarray)->dict:
    """Evalute the model and return the evaluation metrics"""
    try:
        y_pred=clf.predict(X_test)
        y_pred_proba=clf.predict_proba(X_test)[:,1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save(file_path:str,metrics:dict):
    """Saving metrics into json file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path,'w') as f:
            json.dump(metrics,f,indent=4)
        logger.debug('Metrics saved to %s',file_path) 
    except Exception as e:
        logger.error('Some Unexcepted error occurred during the saving process %s',e)
        raise


def main():
    try:
        logger.debug("Process Started")
        clf=load_model(file_path='./models/model.pkl')
        test_data=load_data(file_path='./data/processed/test_tfidf.csv')
        X_test=test_data.iloc[:,:-1]
        y_test=test_data.iloc[:,-1]    
        metrics=evaluate_model(clf,X_test,y_test)
        save(file_path='reports/metrics.json',metrics=metrics)
        logger.debug('All Set')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")


if __name__=="__main__":
    main()        