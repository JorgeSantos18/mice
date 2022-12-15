from typing import Dict, List, Optional
import logging

from allennlp.data import Tokenizer
from overrides import overrides
from nltk.tree import Tree


from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.common.checks import ConfigurationError
from sklearn.model_selection import train_test_split

from pathlib import Path
from itertools import chain
import os.path as osp
import tarfile
import numpy as np
import math

from src.predictors.predictor_utils import clean_text 
logger = logging.getLogger(__name__)

TRAIN_VAL_SPLIT_RATIO = 0.9

        
def get_label(p):
    assert "no" in p or "yes" in p or "partial" in p
    return "0" if "no" in p else "1"

import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
import nltk
from nltk.stem.porter import *
from nltk.stem import *
nltk.download('rslp')
nltk.download('stopwords')
import time
import unidecode
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

class DatasetMgr():

    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def get_prepared_data(self):
        """Function to prepare data"""

        FILE_DECISION_CSV = 'src/predictors/imdb/court_decisions_with_gender.csv'

        # Loading the labeled decisions
        data = pd.read_csv(FILE_DECISION_CSV, sep='|',header=0)
        print('data.shape=' + str(data.shape) + ' full data set')
        # Removing NA values
        data = data.dropna(subset=[data.columns[9]])# decision_description
        data = data.dropna(subset=[data.columns[11]])# decision_label
        print('data.shape=' + str(data.shape) + ' dropna')
        # Removing duplicated samples
        # df = data.groupby(['process_number']).size().reset_index(name='count')
        # print('df.shape=' + str(df.shape) + ' removed duplicated samples by process number')
        data = data.drop_duplicates(subset=[data.columns[1]]) # process_number
        print('data.shape=' + str(data.shape) + ' removed duplicated samples by process_number')
        data = data.drop_duplicates(subset=[data.columns[9]]) # decision_description
        print('data.shape=' + str(data.shape) + ' removed duplicated samples by decision_description')
        # Removing not relevant decision labels and decision not properly labeled
        data = data.query('decision_label != "conflito-competencia"')
        print('data.shape=' + str(data.shape) + ' removed decisions labeled as conflito-competencia')
        data = data.query('decision_label != "prejudicada"')
        print('data.shape=' + str(data.shape) + ' removed decisions labeled as prejudicada')
        data = data.query('decision_label != "not-cognized"')
        print('data.shape=' + str(data.shape) + ' removed decisions labeled as not-cognized')
        data_no = data.query('decision_label == "no"')
        print('data_no.shape=' + str(data_no.shape))
        data_yes = data.query('decision_label == "yes"')
        print('data_yes.shape=' + str(data_yes.shape))
        data_partial = data.query('decision_label == "partial"')
        print('data_partial.shape=' + str(data_partial.shape))
        # Merging decisions whose labels are yes, no, and partial to build the final data set
        data_merged = data_no.merge(data_yes, how='outer')
        data = data_merged.merge(data_partial, how='outer')
        print('data.shape=' + str(data.shape) + ' merged decisions whose labels are yes, no, and partial')
        # Removing decision_description and decision_labels whose values are -1 and -2
        indexNames = data[ (data['decision_description'] == str(-1)) |                    (data['decision_description'] == str(-2)) |                    (data['decision_label'] == str(-1)) |                    (data['decision_label'] == str(-2)) ].index
        # print('indexNames='+str(len(indexNames)))
        data.drop(indexNames, inplace=True)
        print('data.shape=' + str(data.shape) + ' removed -1 and -2 decision descriptions and labels')


        #Stemming leaves only the root of the word. 
        stemmer = PorterStemmer()
        #Create set of stopwords
        stopwords = nltk.corpus.stopwords.words('portuguese')

        start = time.time()
        index = 0
        # data_decision_description_preprocessed=[]

        data['decision_description'] = data['decision_description'].str.lower()
        data['decision_description'] = data['decision_description'].str.replace('.','')
        # i = 0    
        for i in range(len(data['decision_description'])):   
            data['decision_description'][i] = unidecode.unidecode(data['decision_description'][i])
            lineanueva=''
            for pal in data['decision_description'][i].split():
                if pal not in stopwords:
                    lineanueva=lineanueva+stemmer.stem(pal)+" "
            data['decision_description'][i]=lineanueva
            #i += 1

        end = time.time()
        total_time = end - start
        print('Execution time in seconds: ' + str(total_time) )
        print('Execution time in minutes: ' + str(total_time/60) )
        data.head()

        X_data=data[['decision_description']]
        y_data=data['decision_label']

        self.vectorizer.fit(X_data['decision_description'])

        return X_data, y_data
        

    def get_x_tfidf(self,X):
        """Return the TFIDF version of X data"""

        X_tfidf_decision_descr = self.vectorizer.transform(X['decision_description'])

        JudgeGender  = pd.get_dummies(X['judge_gender'],  prefix='judge_gender')
        JudgeGendeValues =JudgeGender.values 
        X_tfidf = hstack((X_tfidf_decision_descr, JudgeGendeValues)).tocsr() 
        return X_tfidf

    def get_y_encoded(self,Y):
        le = preprocessing.LabelEncoder()
        
        y_coded =le.fit_transform(Y)
        return y_coded
    
    def get_y_encoded_without_partial(self,Y):
        y_coded = self.get_y_encoded(Y)    
        y_coded_without_partial = [ (0.0 if sample == 0 else 1.0) for sample in y_coded]
        return y_coded_without_partial
    
    def get_x_tfidf_with_sensitive(self, X):
        X_tfidf_decision_descr = self.vectorizer.transform(X['decision_description'])
        Sensitive = np.where(X["judge_gender"] == "M", 0.0 , 1.0)
        return X_tfidf_decision_descr, Sensitive

    def get_x_tfidf_without_gender(self,X):
        """Return the TFIDF version of X data"""

        X_tfidf_decision_descr = self.vectorizer.transform(X['decision_description'])

        # JudgeGender  = pd.get_dummies(X['judge_gender'],  prefix='judge_gender')
        # JudgeGendeValues =JudgeGender.values 
        # X_tfidf = hstack((X_tfidf_decision_descr, JudgeGendeValues)).tocsr() 
        return X_tfidf_decision_descr


@DatasetReader.register("imdb")
class ImdbDatasetReader(DatasetReader):

    TAR_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz' 
    TRAIN_DIR = 'aclImdb/train'
    TEST_DIR = 'aclImdb/test'

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or \
                {"tokens": SingleIdTokenIndexer()}

        self.random_seed = 0 # numpy random seed

        DataMgr = DatasetMgr()
        X, y = DataMgr.get_prepared_data()
        self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(X, y, test_size=0.3)


    @overrides
    def _read(self, file_path):

      if file_path == 'train':
        for index, row in self.X_train.iterrows():
          yield self.text_to_instance(
          clean_text(row['decision_description'], special_chars=["<br />", "\t"]), 
          get_label(self.y_train[index]))
      else:
        for index, row in self.X_test.iterrows():
          yield self.text_to_instance(
          clean_text(row['decision_description'], special_chars=["<br />", "\t"]), 
          get_label(self.y_test[index]))     
    
    def get_inputs(self, file_path, return_labels = False):
        np.random.seed(self.random_seed)
        
        strings = []
        labels = []

        if file_path == 'train':
          for index, row in self.X_train.iterrows():
            labels.append(get_label(self.y_train[index]))
            strings.append(clean_text(row['decision_description'], 
                                    special_chars=["<br />", "\t"]))
        else:
          for index, row in self.X_test.iterrows():
            labels.append(get_label(self.y_test[index]))
            strings.append(clean_text(row['decision_description'], 
                                    special_chars=["<br />", "\t"]))

        if return_labels:
            return strings, labels
        return strings 


    def text_to_instance(
            self, string: str, label:str = None) -> Optional[Instance]:
        tokens = self._tokenizer.tokenize(string)
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)