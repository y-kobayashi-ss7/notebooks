from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import sys
import pickle

# cur_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
cur_dir = 'C:\\Users\\y-kobayashi\\notebooks\\movieclassifier'
stop = pickle.load(open(os.path.join(cur_dir,'pkl_objects', 'stopwords.pkl'), 'rb'))

def tokenizer(text):
    
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    
    return tokenized

vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)