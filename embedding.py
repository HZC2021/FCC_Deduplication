import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
# from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import spacy
import re
import pickle
import os
from transformers import BigBirdTokenizer, \
BigBirdForSequenceClassification, Trainer, TrainingArguments,EvalPrediction, AutoTokenizer, BigBirdModel

# def get_relations(document):
#     # in our case, relations are bigrams in sentences
#     bigrams = []
#     for sent in document:
#         for i in range(len(sent)-1):
#             # for every word and the next in the sentence
#             pair = [sent[i], sent[i+1]]
#             # only add unique bigrams
#             if pair not in bigrams:
#                 bigrams.append(pair)
#     return bigrams
#
# def preprocess_document(document, sentence_spliter='.', word_spliter=' ', punct_mark=','):
#     # lowercase all words and remove trailing whitespaces
#     document = document.lower().strip()
#
#     # remove unwanted punctuation marks
#     for pm in punct_mark:
#         document = document.replace(pm, '')
#
#     # get list of sentences which are non-empty
#     sentences = [sent for sent in document.split(sentence_spliter) if sent != '']
#
#     # get list of sentences which are lists of words
#     document = []
#     for sent in sentences:
#         words = sent.strip().split(word_spliter)
#         document.append(words)
#
#     return document
#
# def get_entities(document):
#     # in our case, entities are all unique words
#     unique_words = []
#     for sent in document:
#         for word in sent:
#             if word not in unique_words:
#                 unique_words.append(word)
#     return unique_words
#
#
#
# def build_graph(doc):
#     # preprocess document for standardization
#     pdoc = preprocess_document(doc)
#
#     # get graph nodes
#     nodes = get_entities(pdoc)
#
#     # get graph edges
#     edges = get_relations(pdoc)
#
#     # create graph structure with NetworkX
#     G = nx.Graph()
#     G.add_nodes_from(nodes)
#     G.add_edges_from(edges)
#
#     return G




if __name__ == "__main__":
    _DEP_LABELS_DICT = {"acl": "clausal modifier of noun (adjectival clause)",
    "acomp": "adjectival complement",
    "advcl": "adverbial clause modifier",
    "advmod": "adverbial modifier",
    "agent": "agent",
    "amod": "adjectival modifier",
    "appos": "appositional modifier",
    "attr": "attribute",
    "aux": "auxiliary",
    "auxpass": "auxiliary (passive)",
    "case": "case marking",
    "cc": "coordinating conjunction",
    "ccomp": "clausal complement",
    "clf": "classifier",
    "complm": "complementizer",
    "compound": "compound",
    "conj": "conjunct",
    "cop": "copula",
    "csubj": "clausal subject",
    "csubjpass": "clausal subject (passive)",
    "dative": "dative",
    "det": "determiner",
    "discourse": "discourse element",
    "dislocated": "dislocated elements",
    "dobj": "direct object",
    "expl": "expletive",
    "fixed": "fixed multiword expression",
    "flat": "flat multiword expression",
    "goeswith": "goes with",
    "hmod": "modifier in hyphenation",
    "hyph": "hyphen",
    "infmod": "infinitival modifier",
    "intj": "interjection",
    "iobj": "indirect object",
    "list": "list",
    "meta": "meta modifier",
    "neg": "negation modifier",
    "nmod": "modifier of nominal",
    "nn": "noun compound modifier",
    "npadvmod": "noun phrase as adverbial modifier",
    "nsubj": "nominal subject",
    "nsubjpass": "nominal subject (passive)",
    "nounmod": "modifier of nominal",
    "npmod": "noun phrase as adverbial modifier",
    "num": "number modifier",
    "number": "number compound modifier",
    "nummod": "numeric modifier",
    "oprd": "object predicate",
    "obj": "object",
    "obl": "oblique nominal",
    "orphan": "orphan",
    "parataxis": "parataxis",
    "partmod": "participal modifier",
    "pcomp": "complement of preposition",
    "pobj": "object of preposition",
    "poss": "possession modifier",
    "possessive": "possessive modifier",
    "preconj": "pre-correlative conjunction",
    "prep": "prepositional modifier",
    "prt": "particle",
    "quantmod": "modifier of quantifier",
    "rcmod": "relative clause modifier",
    "relcl": "relative clause modifier",
    "reparandum": "overridden disfluency",
    "root": "root",
    "ROOT": "root",
    "vocative": "vocative",
    "xcomp": "open clausal complement"}
    _DEP_LABELS = [i.upper() for i in _DEP_LABELS_DICT.keys()]
    data = pd.read_csv('wait_bk.csv')
    SEQ_LEN = 67





    tokenizer = AutoTokenizer.from_pretrained(r"E:\code\FCC_Transformer\models\tokenizer", local_files_only=True) ## bigbird model used to tokenize


    nlp = spacy.load("en_core_web_trf") ## spacy model used to get the dependency tree
    model = BigBirdModel.from_pretrained(r"e:\code\FCC_Transformer\models\model", local_files_only=True) ## bigbird model used to get the word vectors
    for i in range(1209, len(data) ): #len(data)
        print(i)
        # comment_text = data.iloc[i, 3]
        string = data.iloc[i, 3]
        ## load tokenizer and model

        # string = "\"\t\nDear) sir sir sir sir, \"\t\n.../ robots in\n disguise. I love 10 robots."
        string = re.sub(r"[\([{})\]]", ' ', string)
        string= string.replace("\n", " ")
        string = string.replace("\t", " ")
        string = string.replace("/", " ")
        string = string.replace("\"", " ")
        string = string.replace("-", " ")
        string = string.replace(":", ",")
        string = string.replace("...", ". ")
        string = string.replace("’", "'")
        string = string.replace("\"", " ")
        string = string.replace("“", " ")
        string = string.replace("”", " ")
        string = string.replace("cannot", "can't")
        string = " ".join(string.split())

        print(string)
        # doc = nlpt(string)
        # data = tokenizer(string, padding='max_length', truncation=True, max_length=1024, return_tensors="pt")
        # sentence = 'which you step on to activate it'
        comment_text = string
        sentences_doc = nlp(comment_text)

        dep_sentences = list(sentences_doc.sents)

        dep_sentences_clean = []
        for idx_sentence, dep_sentence in enumerate(dep_sentences):
            if dep_sentence.__len__() <= 2:
                continue
            else:
                dep_sentences_clean.append(dep_sentence)


        SEN_CNT = len(dep_sentences_clean)
        adj_arc_in = np.zeros((SEN_CNT, SEQ_LEN, SEQ_LEN), dtype='int64')
        vec_in = np.zeros((SEN_CNT,SEQ_LEN, 768), dtype='float32')

        nlp = spacy.load("en_core_web_md")
        SEQ_LEN = 67
        V = 768
        SEN_CNT = len(dep_sentences_clean)
        adj_arc_in = np.zeros((SEN_CNT, SEQ_LEN, SEQ_LEN), dtype='int64') ## adjacency matrix
        vec_in = np.zeros((SEN_CNT, SEQ_LEN, V), dtype='float32') ## word vector features
        ## tokenization
        doctf = tokenizer(string, padding = 'max_length', truncation=True, max_length = 1024, return_tensors="pt")
        ## word2vector
        output = model(**doctf, output_hidden_states=True)
        start = 1
        list_start = 0
        for idx_sentence, dep_sentence in enumerate(dep_sentences_clean): ##iter over sentences
            text = dep_sentence.text ## get the sentence
             ## get the start position of the sentence

            for token_i,token in enumerate(dep_sentence.sent): ## iter over words in the sentence
                # if token_i >= SEQ_LEN:
                #     break
                    # SEQ_LEN = token_i

                #     print("1:", token.text)
                    inputs = tokenizer(token.text, padding = 'max_length', truncation=True, max_length = 1024, return_tensors="pt")
                    tmplist = inputs.encodings[0].ids
                    tmplist = np.array(tmplist)
                    step = len(tmplist[tmplist!=0])-2  ## get the length of the tokens for one word
                    if token.text == "n't" or token.text == "'m" or token.text == "'s" or token.text == "'d": ## keep both tokenization same
                        step = 1

                    if token.dep_ == 'ROOT' or token.dep_.upper() in _DEP_LABELS: ## is word?
                        arc_1 = int(token.i) - dep_sentence.start ## get the position of the word in the sentence
                        arc_2 = int(token.head.i) - dep_sentence.start
                        adj_arc_in[idx_sentence,arc_1,arc_2] = 1.0 ## add the arc
                        adj_arc_in[idx_sentence,arc_2, arc_1] = 1.0
                        vec_in[idx_sentence,token.i- dep_sentence.start] = output['last_hidden_state'][0][start].detach().numpy() ## add feature
                        # print("p:",doctf.encodings[0].tokens[start])
                        last1 = token.text
                        last2 = doctf.encodings[0].tokens[start]
                        start += step ## go to next token
                    else:
                        start+=step ## go to next token
        if last2 == "▁"+last1: ## check if the last token is the same
            pass
            np.save("auth/auth_adj_arc_in_%d.npy"%i,adj_arc_in) ## if same, save the data
            np.save("auth/auth_vec_in_%d.npy"%i, vec_in)


pass