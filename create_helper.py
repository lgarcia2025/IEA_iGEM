from huggingface_hub import login
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, PreTrainedTokenizer, EsmForMaskedLM, TrainerCallback
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import numpy as np
import os
#import json
import random
from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from collections import Counter
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from evaluate import load
import simplejson as json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import esm
import py3Dmol

#Helper functions
def iterativeUnmask(masked_sequence, model, tokenizer, num_unmasking_steps=1, temperature=1.0):
    '''
    Iteratively unmasks a sequence. '_' is input as a mask token and converted internally to <mask>. Temperature = 0 gives deterministic unmasking at a given position
    '''
    ESM_ALLOWED_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    inputs = tokenizer(masked_sequence.replace("_", tokenizer.mask_token), return_tensors="pt")
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the model to the appropriate device
    model = model.to(device)
    # Move input tensors to the device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    mask_token_index = tokenizer.mask_token_id
    masked_positions = (inputs['input_ids'] == mask_token_index).nonzero(as_tuple=True)[1]
    predseq = masked_sequence
    if masked_positions.numel() == 0:
        return predseq #Means recursion is finished
    n = int(len(masked_positions) / num_unmasking_steps)
    indices = np.random.choice(masked_positions.cpu().numpy(), size=n, replace=False)
    for index in indices:
        logits = predictions[0, index]
        probabilities = F.softmax(logits, dim=-1)
        # Get top 20 predictions
        top_probs, top_indices = torch.topk(probabilities, tokenizer.vocab_size)
        temp_adjusted_denom = 0
        for prob, idx in zip(top_probs, top_indices):
            #First adjust probabilities according to temperature value. If token in in standard set of amino acids, ignore it.
            predicted_token = tokenizer.decode([idx])
            try:
                if predicted_token in ESM_ALLOWED_AMINO_ACIDS:
                    temp_adjusted_denom += np.power(prob.item(), (1.0/temperature))
            except:
                pass #Handle temp = zero later
        acc_prob = 0.0
        choice = np.random.random() #Pick a number between 0 and 1
        for prob, idx in zip(top_probs, top_indices):
            predicted_token = tokenizer.decode([idx])
            if predicted_token in ESM_ALLOWED_AMINO_ACIDS:
                try:
                    new_prob = prob.item()**(1.0/temperature) / temp_adjusted_denom
                    acc_prob += new_prob
                except:
                    #Means we deterministically choose this amino acid with temp = 0.0
                    predseq = predseq[:index-1] + predicted_token + predseq[index:]
                    break
                if choice < acc_prob:
                    #Means we randomly choose this amino acid
                    predseq = predseq[:index-1] + predicted_token + predseq[index:]
                    break
    return iterativeUnmask(predseq, model, tokenizer, num_unmasking_steps=num_unmasking_steps-1, temperature=temperature)
def betalinear30():
    '''
    Samples from a beta linear distribution
    '''
    r1 = np.random.random()
    if r1 > 0.8:
        #Linear
        return np.random.random()
    else:
        return np.random.beta(a=3, b=9, size=1)[0]
def mask_seq(seq):
    '''
    Takes in an input and applies the betalinear30 function to it, then returns and appropriately masked sequence with it.
    '''
    frac_masked = betalinear30()
    newseq = ''
    for char in seq:
        if np.random.random() < frac_masked:
            #Mask that position
            newseq += '<mask>'
        else:
            newseq += char
    return newseq
def calculate_similarity(alignment):
    matches = sum(a == b for a, b in zip(alignment[0], alignment[1]))
    total = len(alignment[0])
    return matches / total * 100
def tokenize_and_prepare_labels(premasked_fasta_path, unmasked_fasta_path, tokenizer: PreTrainedTokenizer, max_length: int = 2000):
    def read_fasta(file_path):
        sequences = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(str(record.seq))
        return sequences

    # Read sequences from the FASTA files
    premasked_sequences = read_fasta(premasked_fasta_path)
    unmasked_sequences = read_fasta(unmasked_fasta_path)

    # Tokenize sequences
    tokenized_premasked = tokenizer(premasked_sequences, padding="max_length", truncation=True, max_length=max_length)
    tokenized_unmasked = tokenizer(unmasked_sequences, padding="max_length", truncation=True, max_length=max_length)

    # Create labels based on the unmasked sequences
    labels = tokenized_unmasked["input_ids"].copy()

    # Update the labels to ignore special tokens and padding in the masked sequences
    for i in range(len(tokenized_premasked["input_ids"])):
        for j in range(len(tokenized_premasked["input_ids"][i])):
            if tokenized_premasked["input_ids"][i][j] == tokenizer.mask_token_id:
                # Set the label to the corresponding unmasked token
                labels[i][j] = tokenized_unmasked["input_ids"][i][j]
            else:
                # Ignore other tokens
                labels[i][j] = -100

    tokenized_premasked["labels"] = labels
    return tokenized_premasked
def maxSimilarity(seq, seqList):
    '''
    Returns the maximum sequence similarity between a query (seq) and list of sequences (seqList)
    '''
    maxSim = 0
    for seq2 in seqList:
        alignments = pairwise2.align.globalxx(SeqRecord(Seq(seq), id="seq1").seq, SeqRecord(Seq(seq2), id="seq2").seq)
        if calculate_similarity(alignments[0]) > maxSim:
            maxSim = calculate_similarity(alignments[0])
    return maxSim
def evalProt(input_sequences, critic, tokenizer, class_labels = ["Negative", "Positive"]):
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic.to(device)
    inputs = tokenizer(input_sequences, padding=True, truncation=True, return_tensors="pt", max_length=2000)
    # Move inputs to the same device as the model
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    # Run inference
    with torch.no_grad():
        outputs = critic(input_ids=input_ids, attention_mask=attention_mask)
    # Get the logits
    logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Get the predicted class
    predicted_class_ids = torch.argmax(probabilities, dim=-1)
    predicted_labels = [class_labels[i] for i in predicted_class_ids]
    return(predicted_labels, probabilities.cpu().numpy())
def iptm_table(fn, name='outputs/'):
    df = pd.DataFrame(columns = ['Rank', 'iPTM'])
    i = fn.split('rank_00')[1][:1]
    index = fn.split('_scores')[0].split(name)[1]
    try:
        with open(fn, 'r') as f:
            data = json.load(f, strict=False)
        iptm_score = float(data['iptm'])
    except FileNotFoundError:
        print('File not found')
    except json.JSONDecodeError as e:
        print(f'JSON decoding error: {e}')
    except KeyError:
        print("'iptm' key not found in the JSON data")
    except ValueError:
        print("Unable to convert 'iptm' value to float")
    df.loc[len(df)] = [int(i), iptm_score]
    df.index = [index]
    return df
def iterate_json_files(directory, keywords):
    for filename in os.listdir(directory):
        if filename.endswith('.json') and any(keyword.lower() in filename.lower() for keyword in keywords):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                yield file_path
def rearrange(variantName, df):
    vals = list(df.loc[variantName].sort_values(by='Rank')['iPTM'])
    df2 = pd.DataFrame()
    df2.index = [variantName]
    for i, val in enumerate(vals):
        df2[f"iPTM Rank{i}"] = val
    return df2
def get_unique_non_duplicates_with_indices(lst):
    # Count occurrences of each element
    count = Counter(lst)

    # Create a list of tuples (value, index) for unique non-duplicates
    unique_with_indices = [(item, index) for index, item in enumerate(lst) if count[item] == 1]

    # Separate values and indices
    unique_values = [item for item, _ in unique_with_indices]
    unique_indices = [index for _, index in unique_with_indices]

    return unique_values, unique_indices
def read_fasta(filename):
    sequences = []
    names = []
    current_sequence = ""
    current_name = ""

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    sequences.append(current_sequence)
                    current_sequence = ""
                current_name = line[1:].replace('%', '_').split(' ')[0]  # Remove the '>' character
                names.append(current_name)
            else:
                current_sequence += line

        if current_sequence:  # Append the last sequence
            sequences.append(current_sequence)

    return names, sequences
def getIPTMs(directory, fasta_file, keywords=['rank']):
    df = pd.DataFrame()
    for json_file in tqdm(iterate_json_files(directory, keywords)):
        df = pd.concat([df, iptm_table(json_file, name=directory)])
    fin = pd.DataFrame()
    for ele in set(df.index):
        fin = pd.concat([fin, rearrange(ele, df)])
    sequence_names, sequence_list = read_fasta(fasta_file)
    n2 = []
    unique_names, unique_indices = get_unique_non_duplicates_with_indices(sequence_names)
    for i, sequence in enumerate(sequence_list):
        if i in unique_indices:
            n2.append(sequence.split(':')[0])
    df2 = pd.DataFrame(n2, index=unique_names, columns=['Amino Acid Sequence'])
    all_indices = fin.index.union(df2.index)
    fin_reindexed = fin.reindex(all_indices)
    df2_reindexed = df2.reindex(all_indices)
    df = pd.concat([fin_reindexed, df2_reindexed], axis=1, verify_integrity=True)
    df = df.dropna()
    nm = f'{directory}_iptmTable.csv'
    df.to_csv(nm)
    return df
def modify_index(df, prefix):
    df.index = [f"{prefix}_{i}" for i in df.index]
    return df
def getIPTMSeqs(root_dir, name='iPTMSeqs.csv'):
    '''
    Takes in a directory containing subdirectories with 2 items each - an Alphafold directory where json files containing iPTM scores are held and the fasta file containing those co-folded sequences.
    Returns a pandas dataframe of all sequences and AlphaFold Cofolded derived iPTM scores.
    '''
    allData = pd.DataFrame()
    # List all items in the root directory
    for item in os.listdir(root_dir):
        full_path = os.path.join(root_dir, item)
        # Check if the item is a directory
        if os.path.isdir(full_path):
            #print(f"Found directory: {full_path}")
            # List all items in the subdirectory
            for sub_item in os.listdir(full_path):
                sub_full_path = os.path.join(full_path, sub_item)
                if os.path.isdir(sub_full_path):
                    dire = sub_full_path
                if os.path.isfile(sub_full_path):
                    file = sub_full_path
            allData = pd.concat([allData, modify_index(getIPTMs(f'{dire}/', file), item)], axis=0)
    allData.to_csv(name)
    return allData
