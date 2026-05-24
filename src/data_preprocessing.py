import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os 
import logging 
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform(text):
    """Transforms the input text by converting it to lowercase , tokenizing , removing stopwords , punctuation and stemming"""

    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)

def preprocess_df(df,text_column='text', target_column='target'):
    """Preprocessing the  Dataframe by encoding the target columns , removing duplicate amd by transforming them(target_column)"""
    try:
        logger.debug('Starting Preprocessing for Dataframe')

        lb=LabelEncoder()
        df[target_column]=lb.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        df=df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        # Apply the transformation to the specifiled column
        df.loc[:,text_column]=df[text_column].apply(transform)
        logger.debug('Tranformation Done')
        return df
    except KeyError as e:
        logger.error('Columns is not found %s',e)
    except Exception as e:
        logger.error('UnExcception Error during text normalization %s',e)


def save(train_processed:pd.DataFrame , test_processed:pd.DataFrame , folder_name:str):
    """Saving the test and train preprocessed data into the data folder """
    try:
        data_path='data'
        os.makedirs(data_path,exist_ok=True)

        save_path=os.path.join(data_path,folder_name)
        os.makedirs(save_path,exist_ok=True)

        train_processed.to_csv(os.path.join(save_path, "train_processed.csv"), index=False)
        test_processed.to_csv(os.path.join(save_path, "test_processed.csv"), index=False)

        logger.debug('All file Saved Sucessfully')
    
    except Exception as e:
        logger.error('Some error in saving the file %s',e)

def main():
    try:
        test_data=pd.read_csv(r'.\data\raw\test.csv')       
        
        train_data=pd.read_csv(r'.\data\raw\train.csv')

        train_processed=preprocess_df(train_data) 
        
        test_processed=preprocess_df(test_data) 

        save(train_processed,test_processed,folder_name='interim')

        logger.debug('All Set')
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")  


if __name__ == '__main__':
    main()






