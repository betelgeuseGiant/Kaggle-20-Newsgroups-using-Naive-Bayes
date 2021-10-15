# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 13:35:47 2018

@author: Karthik Kumarsubramanian
"""

import numpy as np
from os import listdir
from os.path import join

def test_train_split(root):
    
    ## Reading the folders inside the train and test respectively
    subfolder = [folders for folders in listdir(root)]
    
    for f in subfolder:
        if f == "Train":
            fpath = join(root, f)
            ## print(fpath)
            folder_train = [folders for folders in listdir(fpath)]
        else:
            fpath = join(root, f)
            ## print(fpath)
            folder_test = [folders for folders in listdir(fpath)]
            
    ## print(type(folder_train))
    ## print(len(folder_test))
    ## print(subfolder)
    
    ## Reading the folders and creating 2D list of all files
    
    files_train = []
    files_test = []
    
    for i in range(len(subfolder)):
        if(subfolder[i] == "Train"):
            for j in folder_train:
                fpath = join(root, join(subfolder[i], j))
                files_train.append([f for f in listdir(fpath)])
        else:
            for j in folder_test:
                fpath = join(root, join(subfolder[i], j))
                files_test.append([f for f in listdir(fpath)])    
    
    ## print(sum(len(files_train[i]) for i in range(len(folder_train))))
    ## print("Train: ", files_train)
    
    ## Creating list of folder paths for train and test respectively
    
    path_train = []
    path_test = []
    
    for i in range(len(subfolder)):
        if(subfolder[i] == "Train"):
            for j in range(len(folder_train)):
                for k in files_train[j]:
                    path_train.append(join(root, join(subfolder[i], join(folder_train[j], k))))
        else:
            for j in range(len(folder_test)):
                for k in files_test[j]:
                    path_test.append(join(root, join(subfolder[i], join(folder_test[j], k))))
    
    ## print(len(path_train))
    ## print(len(path_test))
    
    y_train = []
    y_test = []
    
    for i in range(len(subfolder)):
        if(subfolder[i] == "Train"):
            for j in folder_train:
                fpath = join(root, join(subfolder[i], j))
                ## print("Train: ", fpath)
                
                no_of_files = len(listdir(fpath))
                for k in range(no_of_files):
                    y_train.append(j)
        else:
            for j in folder_test:
                fpath = join(root, join(subfolder[i], j))
                ## print("Test: ", fpath)
                
                no_of_files = len(listdir(fpath))
                for k in range(no_of_files):
                    y_test.append(j)
    
    ## print(len(y_train))
    ## print(len(y_test))
    
    words_train = []
    for i in path_train:
        words_train.append(tokenize(i))
    ## print(len(words_train))
    
    p_words = []
    
    for i in words_train:
        for j in i:
            p_words.append(j)
    ## print(len(p_words))
    
    words_array = np.asarray(p_words)
    ## print(words_array.size)
    
    unique_words_train, count = np.unique(words_array, return_counts= True)
    ## print(len(wrds))
    
    ## Creating dictionary for training words and count
    diction_train = {}
    num = 1
    
    for word in words_train:
        
        words_array = np.asarray(word)
        w, count = np.unique(words_array, return_counts= True)
        diction_train[num] = {}
        
        for i in range(len(w)):
            diction_train[num][w[i]] = count[i]
        num = num + 1
    
    ## Creating 2d array of unique words and count    
    X_train = []
    for i in diction_train.keys():
        r = []
        for w in unique_words_train:
            if(w in diction_train[i].keys()):
                r.append(diction_train[i][w])
            else:
                r.append(0)
                
        X_train.append(r)
        
    X_train = np.asarray(X_train)
    
    words_test = []
    
    for i in path_test:
        words_test.append(tokenize(i))
    ## print(len(words_test))
    
    t_words = []
    
    for i in words_test:
        for j in i:
            t_words.append(j)
    ## print(len(t_words))
    words_array_test = np.asarray(t_words)
    ## print(words_array_test.size)
    
    diction_test = {}
    num = 1
    
    for word in words_test:
        
        words_array_test = np.asarray(word)
        w, count = np.unique(words_array_test, return_counts= True)
        diction_test[num] = {}
        
        for i in range(len(w)):
            diction_test[num][w[i]] = count[i]
        num = num + 1
    
    unique_words_test, count = np.unique(words_array, return_counts= True)
    
    X_test = []
    for i in diction_test.keys():
        r = []
        for w in unique_words_test:
            if(w in diction_test[i].keys()):
                r.append(diction_test[i][w])
            else:
                r.append(0)
                
        X_test.append(r)
        
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test =  np.asarray(y_test)
    
    return X_train, X_test, y_train, y_test, diction_test, unique_words_train, unique_words_test


stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "youre", "youve", "youll", "youd", 
             'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "shes", 'her', 'hers', 
             'herself', 'it', "its", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
             'who', 'whom', 'this', 'that', "thatll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
             'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
             'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
             'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
             'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
             'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
             'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "dont", 'should', "shouldve", 'now', 'd', 'll', 'm',
             'o', 're', 've', 'y', 'ain', 'aren', "arent", 'couldn', "couldnt", 'didn', "didnt", 'doesn', "doesnt", 'hadn', 
             "hadnt", 'hasn', "hasnt", 'haven', "havent", 'isn', "isnt", 'ma', 'mightn', "mightnt", 'mustn', "mustnt", 
             'needn', "neednt", 'shall', "shallnt", 'shouldn', "shouldnt", 'wasn', "wasnt", 'weren', "werent", 'won', "wont", 
             'wouldn', "wouldnt"]

def formatting(words):
    
    ## Removing tabs, special characters and punctuations
    p_words = []
    for word in words:
        p_words.append(''.join(i for i in word if i.isalnum()))
        
    words = p_words.copy()
    
    ## Removing alphanumeric words
    words = [i for i in words if not i.isdigit()]
    
    ## Removing words of length 1
    words = [i for i in words if not len(i) == 1]
    
    ## Removing extra blanks from the words 
    words = [str for str in words if str]

    ## Converting all words to lower case
    words = [i.lower() for i in words]
    
    ## Removing words with length less 2
    words = [i for i in words if len(word) > 2]
    
    return words

def tokenize(path):
    
    file = open(path, 'r')
    
    ## Reading the sentences inside the file
    lines = file.readlines()
    
    ## Removing metadata
    p = 0
    for i in range(len(lines)):
        if(lines[i] == '\n'):
            p = i + 1
            break
            
    text = lines[p:]
    
    ## Breaking the sentences into words
    words = []
    for line in text:
        words = line[0:len(line)-1].strip().split(" ")
    
    ## Preprocessing the words void of tabs, punctuations and special character
    words = formatting(words)
    
    ## Removing stopwords from the list of stopwords
    words = [i for i in words if not i in stopwords]
    
    return words

def create_dict(X_train, Y_train, features):
    
    ## Result dictionary maps words and count in X_train to the corresponding Y_train
    result = {}
    
    ## Gets unique classes in the Y_train and count of the classes
    classes, count = np.unique(Y_train, return_counts=True)
    
    for i in range(len(classes)):
        current_class = classes[i]
        
        ## Creating item "Total data" for probability calculation
        result["Total_classes"] = len(Y_train)
        result[current_class] = {}
        
        X_train_current = X_train[Y_train == current_class]
        
        ## Number of unique words in the X_train
        num_features = len(features)
        
        for j in range(num_features):
            
            ## Gets the total count of the word in the class
            result[current_class][features[j]] = X_train_current[:,j].sum() 
                
        result[current_class]["Total_word_count"] = count[i]
    
    return result


def log_probablity(result, x, current_class):
    
    output = np.log(result[current_class]["Total_word_count"]) - np.log(result["Total_classes"])
    
    num = len(x)
    
    for i in range(num):
        
        if(x[i] in result[current_class].keys()):
            
            xi = x[i]
            count_current_class_equal_xi = result[current_class][xi] + 1
            count_curr_class = result[current_class]["Total_word_count"] + len(result[current_class].keys())
            curr_xj_prob = np.log(count_current_class_equal_xi) - np.log(count_curr_class)
            output = output + curr_xj_prob
        
        else:
            continue
    
    return output

def predict(result, X_test):
    
    y_pred = []
    
    for x in X_test:
        
        classes = result.keys()
        p = -10000
        nearest_class = -1
        
        for curr_class in classes:
           
            if(curr_class == "Total_classes"):
                continue
            
            p_curr_class = log_probablity(result, x, curr_class)
            
            if(p_curr_class > p):
               
                p = p_curr_class
                nearest_class = curr_class
            
        y_predicted = nearest_class
        y_pred.append(y_predicted)
    
    return y_pred

def find_accuracy(predicted_classes, y_test):
    
    count = 0
    for i in range(len(y_test)):
        
        if predicted_classes[i] == y_test[i]:
            
            count = count + 1
            
        else:
            
            continue
        
    accuracy = (count/len(y_test)) * 100
    
    return accuracy

root = "20_newsgroups"
X_train, X_test, y_train, y_test, test_dict, uw_train, uw_test = test_train_split(root)

## print(len(X_train), len(X_test), len(y_train), len(y_test))

result = create_dict(X_train, y_train, uw_train)
## print(result)

X_test_keys = []
for i in test_dict.keys():
    X_test_keys.append(list(test_dict[i].keys()))
    
predicted_classes = np.asarray(predict(result, X_test_keys))
## print(len(predicted_classes))

acc = find_accuracy(predicted_classes, y_test)
print("\n Accuracy is: ", acc)
