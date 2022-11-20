import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
# from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
import numpy as np
import re
import pickle
import datasets

if __name__ == "__main__":
    data = pd.read_csv('all_random.csv') ## load comments set
    ## get the text in data
    for i in range(data.shape[0]): # dataset cleaning
        string = data['text_data'][i]
        string = re.sub(r"[\([{})\]]", ' ', string)
        string = string.replace("\n", " ")
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
        data['text_data'][i] = string

    data.to_csv("all_random_cleaned.csv", index=False)
    trainid = np.load("trainid.npy") ##load same training samples as in the GCN model
    valid = np.load("valid.npy")
    testid = np.load("testid.npy")
    data.iloc[trainid].to_csv("train.csv", index=False)
    data.iloc[valid].to_csv("val.csv", index=False)
    data.iloc[testid].to_csv("test.csv", index=False)
    # traindata = datasets.load_dataset('csv', data_files='train.csv')['train']
    # valdata = datasets.load_dataset('csv', data_files='val.csv')['train']
    # testdata = datasets.load_dataset('csv', data_files='test.csv')['train']
    pass



