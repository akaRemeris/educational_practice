import re
import collections

REG_EXP = r'[\w\d]+'
regex_tokenizer = re.compile(REG_EXP, re.I)

def ngram_tokenizer(document, gram_order=2):
    return [document[gram_start:gram_start+gram_order] for gram_start in range(len(document) - gram_order + 1)]

def basic_tokenizer(text_piece, token_min_len=4, regexer=regex_tokenizer, word_normalizer=None):
    regexed = regexer.findall(text_piece.lower())
    
    if word_normalizer is None:
        return [token for token in regexed if len(token) > token_min_len]
    else:
        return [word_normalizer(token) for token in regexed if len(token) > token_min_len]

def tokenize_corprus(text, tokenizer=basic_tokenizer, **tokenizer_args):
    return [tokenizer(doc, **tokenizer_args) for doc in text]

def build_vocabulary(tokenized_corpus, max_token_freq=0.8, min_token_count=5, fake_token=None):
    token_counter = collections.defaultdict(int)
    document_counter = 0
    for doc in tokenized_corpus:
        document_counter += 1
        for unique_token in set(doc):
            for token in doc:
                if unique_token == token:
                    token_counter[unique_token] += 1    
    
    filtered_token_dict = {token: n_entries for token, n_entries in token_counter.items()
                                            if n_entries/document_counter <= max_token_freq
                                            and n_entries >= min_token_count}

    sorted_keys = sorted(filtered_token_dict, key=lambda x: (token_counter[x], x), reverse=False)
    if fake_token is not None:
        sorted_keys.insert(0, fake_token)
    vocabulary = collections.OrderedDict(zip(sorted_keys, [i for i in range(len(sorted_keys))]))
    token_freq = [token_counter[token]/document_counter for token in sorted_keys]
    return vocabulary, token_freq

def corp_encode(tokenized_corp, vocabulary):
    corp = list()
    for doc in tokenized_corp:
        corp.append([vocabulary[token] for token in doc if token in vocabulary])
    # unreadable: return [[vocabulary[token] for token in doc if token in vocabulary] for doc in tokenized_corp[:10]]
    return corp