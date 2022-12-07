import imp
import numpy as np
import scipy.sparse

class Vectorizer(object):
    def __init__(self, vocabulary, token_freq):
        self.vocabulary = vocabulary
        self.token_freq = np.array(token_freq)
        self.normalization_parameters = []
    
    def get_tf(self, tokenized_corpus):
        # count_term(term, doc)/size(doc)
        features_shape = (len(tokenized_corpus), len(self.vocabulary))
        term_counter = scipy.sparse.dok_matrix(features_shape, dtype=np.float32)

        for doc_idx, doc in enumerate(tokenized_corpus):
            for token in doc:                
                if token in self.vocabulary:
                    term_counter[doc_idx, self.vocabulary[token]] += 1

        self.term_counter = term_counter.copy()
        tf_features = term_counter.tocsr()
        tf_features = tf_features.multiply(1/tf_features.sum(axis=1))
        return tf_features.log1p()
    
    def get_df(self):
        #N_token_containing_docs/corp_size
        return self.token_freq
    
    def get_tfidf(self, tokenized_corpus, training=True):
        tf_features = self.get_tf(tokenized_corpus)
        df_features = self.get_df()
        idf_features = np.log(1/df_features)
        tfidf_features = tf_features.tocsr()
        tfidf_features = tfidf_features.multiply(idf_features)
        # add parameter for normalizer function
        tfidf_features = tfidf_features.tocsc()
        if training:
            self.normalization_parameters.append(tfidf_features.max() + 1e-6)
        tfidf_features -= tfidf_features.min()
        tfidf_features /= self.normalization_parameters[-1]
        return tfidf_features.tocsr()

class SparseDataLoader(object):
    def __init__(self, sparse_features):
        self.sparse_features = sparse_features
        self.shape = sparse_features.shape
    
    def __len__(self):
        return self.sparse_features.shape[0]
    
    def __getitem__(self, idx):
        requested_features = self.sparse_features[idx].toarray()
        return requested_features