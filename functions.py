#!/usr/bin/env python
# coding: utf-8
import numpy as np      
import matplotlib.pyplot as plt 
import scipy.io.wavfile 
import subprocess
import librosa
import librosa.display
import IPython.display as ipd

from pathlib import Path, PurePath   
from tqdm.notebook import tqdm

import random
import pickle
import pandas as pd
from collections import defaultdict

import utility

N_TRACKS = 1413
HOP_SIZE = 512
OFFSET = 1.0
DURATION = 30
THRESHOLD = 0 # TODO: to be tuned!

def zeros_ones_matrix(m, file_name, tracks_list):
    '''
    INPUT
    m: upper bound, maximum index that can be associated with a peak in the track
    tracks_list: list of tracks (saved in the main folder; settings section of the notebook) 
                eg: (0, WindowsPath('data/mp3s-32k/aerosmith/Aerosmith/01-Make_It.wav'))
    file_name: file sub-name to save
    nh, c_0, c_1, p, n: check above
    OUTPUT
    tracks_matrix: tracks matrix (m, N_TRACKS); each column k corresponds to a track; for each track non-zero entries (ones) are associated  with the indexes in which the peaks occur 
    '''
    N_TRACKS = len(tracks_list)
    
    # inizialize zero matrix
    tracks_matrix = np.zeros((m, N_TRACKS))
    
    # for each track we assign value 1 to the indexes in which peaks occur 
    for k in tqdm(range(len(tracks_list))):
        _, _, _, peaks = utility.load_audio_peaks(tracks_list[k][1], OFFSET, DURATION, HOP_SIZE)
        
        tracks_matrix[peaks, k] = 1
    
    # store tracks matrix of 'int8' items
    file = open(f"{file_name}_matrix.pkl", "wb")
    pickle.dump(tracks_matrix.astype('int8'), file)
    file.close()
    
    return tracks_matrix
    

def minhash(c_0, c_1, p, n, x):
    '''
    INPUT
    c_0, c_1: (nh, 1) arrays of hash functions coefficients randomly generated (nh: number of hash functions)
    p: prime number greater then upper bound m 
    n: upper bound of the target set
    x: array of indexes of peaks
    OUTPUT
    signature: track signature (nh, 1)
    '''
    # We apply the $i^{th}$ hash function to each component of x.
    # The $i^{th}$ component of the signature is equal to the minimum among these values. 
    signature = (((np.outer(c_1, x)+(np.ones((len(x), 1))*c_0).transpose())%p)%n).min(axis=1)
    
    return signature

def flatten(seq):
    l = []
    for elt in seq:
        t = type(elt)
        if t is tuple or t is list:
            for elt2 in flatten(elt):
                l.append(elt2)
        else:
            l.append(elt)
    return l

def generate_signature(tracks_list, tracks_matrix, file_name, nh, c_0, c_1, p, n):
    '''
    INPUT
    tracks_list: list of tracks (saved in the main folder; settings section of the notebook) 
                eg: (0, WindowsPath('data/mp3s-32k/aerosmith/Aerosmith/01-Make_It.wav'))
    file_name: file sub-name to save
    tracks_matrix, nh, c_0, c_1, p, n: check above
    OUTPUT
    signature_matrix: signature matrix (nh, N_TRACKS) 
    '''
    # load track list
    open_file = open(f"{file_name}_list.pkl", "rb")
    tracks_list = pickle.load(open_file)
    open_file.close()
    
    # inizialize matrix
    signature_matrix = np.zeros((nh, len(tracks_list)))
    
    # apply minhash function to all the tracks and update signature matrix each time
    for k in tqdm(range(len(tracks_list))):
        # generate a list of indexes of non-zero entries from the $k^{th}$ column of track_matrix  
        peaks = flatten(np.argwhere(tracks_matrix[:,k]).tolist())
        
        signature_matrix[:,k] = minhash(c_0, c_1, p, n, peaks)
    
    # store signature matrix of 'int8' items
    file = open(f"signature_matrix_{file_name}.pkl", "wb")
    pickle.dump(signature_matrix.astype('int8'), file)
    file.close()
    
    return signature_matrix

def lsh(bands, signature_query, signature_matrix):
    '''
    INPUT
    bands: number of bands
    signature_matrix: signature matrix of tracks
    signature_query: signature matrix of queries
    OUTPUT
    candidate_pairs: list of lists; each sub-list is a list of candidates indexes that share at least one bucket with a query
    '''
    p2 = 1549 # prime number greater then the upper bound (n) of minhashing  target set
    nh = signature_query.shape[0]
    # generate nh/bands random coefficients for each band 
    coeff_matrix = np.array(np.random.randint(0, p2-1, size = (int(nh/bands), bands)))
    # split signature_query and signature_matrix into bands
    split_query = np.split(signature_query, bands, axis=0)
    split_matrix = np.split(signature_matrix, bands, axis=0)
    
    # inizialize candidate dictionary
    # key: query index
    # value: (number of bands) sets of indexes of tracks sharing the bucket with query in a specific band
    candidate = defaultdict()
    for w in range(signature_query.shape[1]):
        candidate[w] = []
        
    for i in range(bands):
        # inizialize i^th band dictionary
        # key: bucket
        # value: track index
        buckets = defaultdict()
        coeff = coeff_matrix[:,i]
        def lsh_hash(signature):
            # hash band signature to bucket
            bucket = np.inner(coeff, signature) % p2
            return bucket
        buckets['coefficients'] = coeff
        hash_query = np.apply_along_axis(lsh_hash, 0, split_query[i])
        hash_band = np.apply_along_axis(lsh_hash, 0, split_matrix[i])
        for j in range(len(hash_band)):
            if hash_band[j] not in buckets.keys():
                buckets[hash_band[j]] = [j]
            else:
                buckets[hash_band[j]].append(j)

        for k in range(len(hash_query)):
            candidate[k].append(set(buckets[hash_query[k]]))
        
        # store i^th band dictionary
        open_file = open(f"{bands}_bands/band_{i+1}.pkl", "wb")
        pickle.dump(buckets, open_file)
        open_file.close()
        
    print(candidate)    
    candidate_pairs = []
    
    for v in range(len(candidate)):
        # append to candidate_pairs list a list of candidates indexes that share at least one bucket with the v^th query
        candidate_pairs.append(list(set.union(*candidate[v])))
    
    return candidate_pairs

def find_matches(signature_matrix, signature_query, candidate, tracks_titles, threshold):
    '''
    INPUT
    signature_matrix: signature matrix of tracks
    signature_query: signature matrix of queries
    candidate: list of lists; each sub-list is a list of candidates indexes that share at least one bucket with a query (output of lsh function)
    tracks_titles: list of tracks features; eg. (0, ['01-Make It', 'aerosmith'])
    threshold: jaccard similarity threshold
    
    OUTPUT
    similar: list of lists; each sub-list is a list (associated with a query) of lists [match index, jaccard similarity] 
    '''
    nh = signature_query.shape[0]
    similar = []
    # clean titles
    new_col = [item[1][0].split("-", 1)[1] for item in tracks_titles]
    for j in range(signature_query.shape[1]):
        query = signature_query[:,j]
        similar.append([])
        # boolean check of jaccard similarity of the j^th query signature with its candidates signatures
        for i in candidate[j]:
            jac = len(query[query == signature_matrix[:,i]]) / nh
            if jac > threshold:
                similar[-1].append([i, round(jac,1)])
                if j == 0:
                    print('The','\033[1m' + f'{j+1}st' + '\033[0m', f'track you are looking for could be', '\033[1m' + f'{new_col[i]}, sung by {tracks_titles[i][1][1].title()}.' + '\033[0m', f'Jaccard similarity is equal to {round(jac,1)}.')
                    print()
                elif j == 1:
                    print('The','\033[1m' + f'{j+1}nd' + '\033[0m', f'track you are looking for could be', '\033[1m' + f'{new_col[i]}, sung by {tracks_titles[i][1][1].title()}.' + '\033[0m', f'Jaccard similarity is equal to {round(jac,1)}.')
                    print()
                else:
                    print('The','\033[1m' + f'{j+1}th' + '\033[0m', f'track you are looking for could be', '\033[1m' + f'{new_col[i]}, sung by {tracks_titles[i][1][1].title()}.' + '\033[0m', f'Jaccard similarity is equal to {round(jac,1)}.')
                    print()
    return similar

def find_similarities(signature_matrix, signature_query, tracks_titles, threshold):
    '''
    as find_matches function, except for jaccard similarity check
    '''
    nh = signature_query.shape[0]
    similar = []
    # clean titles
    new_col = [item[1][0].split("-", 1)[1] for item in tracks_titles]
    for j in range(signature_query.shape[1]):
        query = signature_query[:,j]
        similar.append([])
        # boolean check of jaccard similarity of the j^th query signature with all track signatures
        for i in range(signature_matrix.shape[1]):
            jac = len(query[query == signature_matrix[:,i]]) / nh
            if jac > threshold:
                similar[-1].append([i, round(jac,1)])
                if j == 0:
                    print('The','\033[1m' + f'{j+1}st' + '\033[0m', f'track you are looking for could be', '\033[1m' + f'{new_col[i]}, sung by {tracks_titles[i][1][1].title()}.' + '\033[0m', f'Jaccard similarity is equal to {round(jac,1)}.')
                    print()
                elif j == 1:
                    print('The','\033[1m' + f'{j+1}nd' + '\033[0m', f'track you are looking for could be', '\033[1m' + f'{new_col[i]}, sung by {tracks_titles[i][1][1].title()}.' + '\033[0m', f'Jaccard similarity is equal to {round(jac,1)}.')
                    print()
                else:
                    print('The','\033[1m' + f'{j+1}th' + '\033[0m', f'track you are looking for could be', '\033[1m' + f'{new_col[i]}, sung by {tracks_titles[i][1][1].title()}.' + '\033[0m', f'Jaccard similarity is equal to {round(jac,1)}.')
                    print()
    return similar
