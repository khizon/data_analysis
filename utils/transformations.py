from sklearn.base import BaseEstimator, TransformerMixin
import re
import nltk
from ntlk.corpus import stopwords
nltk.download('stopwords')

class CleanText(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.words_to_remove = []
        languages = ['english', 'french', 'spanish', 'norwegian', 'swedish', 'german']

        for language in languages:
            self.stopwords = set(stopwords.words(language))
            self.words_to_remove.extend(self.stopwords)

        self.regex_patterns = {
            'new line': re.compile(r'_x000d_\n'),
            'field_names': re.compile(r'.*?:\s'),
            'phone_number': re.compile(r'\+?[0-9]{2,3}[-\s]?[0-9]{3}[-\s]?[0-9]{3,4}[-\s]?[0-9]{3,4}'),
            'email_address': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}'),
            'website_url': re.compile(r'(https?://)?(www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,4}(/[A-Za-z0-9._%+-]*)*'),
            'html_code': re.compile(r'<([A-Za-z0-9]+)(\s[A-Za-z0-9]+="[^"]*")*>(.*?)</\1>'),
            'non_alphabetical_symbols': re.compile(r'[^a-zA-Z\s]')
        }

    def normalize_text(self, text):
        if isinstance(text, str):
            text = text.lower()

            for k, pattern in self.regex_patterns.items():
                if k == 'new line':
                    text = pattern.sub(' ', text)
                else:
                    text = pattern.sub('', text)
            if len(self.words_to_remove) > 0:
                text = ' '.join(word for word in text.split() if word not in self.words_to_remove)

        else:
            text = ''

        return text
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_t = X.copy()
        X_t = X_t.astype(str)
        X_t['Concat_Text'] = X_t.agg(' '.join, axis=1)
        X_t = X_t.apply(self.normalize_text)
        return X_t
    
class ConcatText(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_t = X.copy()
        X_t = X_t.astype(str)
        X_t['Concat_Text'] = X_t.agg(' '.join, axis=1)
        return X_t
    
class SplitDateTimeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_t = X.copy()

        cols = X_t.select_dtypes(include=['datetime64']).columns.to_list()
        for col in cols:
            for component in ['year', 'month', 'day', 'hour', 'dayofweek']:
                new_col_name = f'{col}_{component}'
                X_t[new_col_name] = getattr(X_t[col].dt, component).astype(str)
        X_t.drop(columns=cols, inplace=True)

        return X_t