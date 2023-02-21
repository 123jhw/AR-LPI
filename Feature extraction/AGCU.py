#!/usr/bin/python3 
from hyperparams import *
from copy import deepcopy
import numpy as np
#!/usr/bin/python3 
isPrint = False
###############################################################################
# fasta methods
###############################################################################
from Bio import SeqIO

def read_fasta_file(fpath):
    fasta_sequences = SeqIO.parse(open(fpath),'fasta')
    seq_dict = {}
    for i, fasta in enumerate(fasta_sequences):
        name, sequence = fasta.id, str(fasta.seq)
        if isPrint : print(name, sequence)
        seq_dict[name] = sequence        
    return seq_dict

def read_RPI_fasta(size):
    rna_seq_path = SEQ_PATH["RPI"][size]["RNA"]
    rna_seqs = read_fasta_file(rna_seq_path)
    protein_seq_path = SEQ_PATH["RPI"][size]["Protein"]
    protein_seqs = read_fasta_file(protein_seq_path)
    return rna_seqs, protein_seqs

###############################################################################
# Pair methods
###############################################################################

def read_pair_file(fpath):
    f = open(fpath, "r")
    flines = f.readlines()
    pairs = []
    for line in flines:
        line = line.replace("\n","")
        p1, p2, label = line.split("\t")
        if isPrint : 
            print("P1: {} / P2: {} / Label: {}".format(p1,p2,label))
        
        pairs.append((p1,p2,label))
    return pairs

def read_RPI_pairs(size):
    pair_path = PAIRS_PATH["RPI"][size]
    pairs = read_pair_file(pair_path)
    return pairs




###############################################################################
# Pair-Seq methods
###############################################################################

def read_RPI_pairSeq(size):
    X, Y = [], []
    pairs = read_RPI_pairs(size)
    rseq, pseq = read_RPI_fasta(size)
    for protein_id, rna_id, label in pairs:
        X.append([pseq[protein_id], rseq[rna_id]])
        Y.append(int(label))
    
    return X, Y
#if __name__ == "__main__":
    # Example
 #   read_RPI_pairSeq(369)



isPrint = True

# Reduced Protein letters(7 letters)
def get_reduced_protein_letter_dict():
    rpdict = {}
    reduced_letters = [["A","G","V"],
                       ["I","L","F","P"],
                       ["Y","M","T","S"],
                       ["H","N","Q","W"],
                       ["R","K"],
                       ["D","E"],
                       ["C"]]
    changed_letter = ["A","B","C","D","E","F","G"]
    for class_idx, class_letters in enumerate(reduced_letters):
        for letter in class_letters:
            rpdict[letter] = changed_letter[class_idx]
    
    return rpdict

# Improved CTF 
class improvedCTF:
    def __init__(self, letters, length):
        self.letters = letters
        self.length = length
        self.dict = {}
        self.generate_feature_dict()
        
    def generate_feature_dict(self):
        def generate(cur_key, depth):
            if depth == self.length:
                return
            for k in self.letters:
                next_key = cur_key + k
                generate(next_key, depth+1)
                if depth == self.length-1:
                    self.dict[next_key] = 0
                    
                
        generate(cur_key="",depth=0)
        
        if isPrint:
            print("iterate letters : {}".format(self.letters))
            print("number of keys  : {}".format(len(self.dict.keys())))
            #print("iterate letters : {}".format(self.dict.keys()))    
        
    
    def get_feature_dict(self):
        for k in self.dict.keys():
            self.dict[k] = 0
            
        return deepcopy(self.dict)

    
# CTF feature processing
def preprocess_feature(x, y, npz_path):
    
    def min_max_norm(a):
        a_min = np.min(a)
        a_max = np.max(a)
        return (a - a_min)/(a_max - a_min)
    
    rpdict = get_reduced_protein_letter_dict()
    feature_x = []
    r_mer = 3
    r_CTF = improvedCTF(letters=["A","C","G","U"],length=r_mer)
    #r_feature_dict = r_CTF.get_feature_dict()
    
    p_mer = 3
    p_CTF = improvedCTF(letters=["A","B","C","D","E","F","G"],length=p_mer)
    #p_feature_dict = p_CTF.get_feature_dict()
    
    x_protein = []
    x_rna = []
        
    for idx, (pseq, rseq) in enumerate(x):
        
        r_feature_dict = r_CTF.get_feature_dict()
        p_feature_dict = p_CTF.get_feature_dict()
        rpseq = []
        for p in pseq:
            if p=="X": 
                rpseq.append(p)
            else:
                rpseq.append(rpdict[p])
                
        pseq = rpseq
        temp_pseq = ""
        for p in pseq:
            temp_pseq += p
        pseq = temp_pseq
        
        for mer in range(1,p_mer+1):
            for i in range(0,len(pseq)-mer):
                pattern = pseq[i:i+mer]
                try:
                    p_feature_dict[pattern] += 1
                except:
                    continue
                #print(pattern)
        Pw = np.array([pseq.count("A")/len(pseq),pseq.count("B")/len(pseq),pseq.count("C")/len(pseq),pseq.count("D")/len(pseq),pseq.count("E")/len(pseq),pseq.count("F")/len(pseq),pseq.count("G")/len(pseq)])
        #print(Pw)
        #p_array = p_feature_dict.keys()
        p_E = {}
        for i in range(0,7):
            Ep1 = Pw[i]
            for j in range(0,7):
                Ep2 = Ep1*Pw[j]
                for k in range(0,7):
                    z = i*49+j*7+k
                    p_E[z] = Ep2*Pw[k]*len(pseq)
      

        
        for mer in range(1,r_mer+1):
            for i in range(0,len(rseq)-mer):
                pattern = rseq[i:i+mer]
                try:
                    r_feature_dict[pattern] += 1
                except:
                    continue
                #print(pattern)
        Rw = np.array([rseq.count("A")/len(rseq),rseq.count("C")/len(rseq),rseq.count("G")/len(rseq),rseq.count("U")/len(rseq)])
        #print(Rw)
        r_E = {}
        for i in range(0,4):
            Er1 = Rw[i]
            for j in range(0,4):
                Er2 = Er1*Rw[j]
                for k in range(0,4):
                    z = i*49+j*7+k
                    r_E[z] = Er2*Rw[k]*len(rseq)

        
        
        p_feature = np.array(list(p_feature_dict.values()))
        p_E = np.array(list(p_E.values()))
        p_feature = p_feature-p_E
        #p_feature = min_max_norm(p_feature)
        #print(p_E)
        r_feature = np.array(list(r_feature_dict.values()))
        #r_feature = min_max_norm(r_feature)
        r_E = np.array(list(r_E.values()))
        r_feature = r_feature-r_E
        
        x_protein.append(p_feature)
        x_rna.append(r_feature)
        
        if isPrint : 
            print("CTF preprocessing ({} / {})".format(idx+1, len(x)))
            #print(r_feature)
            
                
    
    x_protein = np.array(x_protein)
    x_rna = np.array(x_rna)
    y = np.array(y)
    np.savez(npz_path,XP=x_protein, XR=x_rna, Y=y)
    
    if isPrint :
        print("Protein feature : {}".format(x_protein.shape))
        print("RNA feature     : {}".format(x_rna.shape))
        print("Labels          : {}".format(y.shape))
        print("Saved path      : {}".format(npz_path))
    
     
    return x_protein, x_rna, y

def preprocess_and_savez_NPInter():
    X, Y = read_NPInter_pairSeq()
    XP, XR, Y = preprocess_feature(X, Y, NPZ_PATH["NPInter"])
    
def preprocess_and_savez_RPI(size):
    X, Y = read_RPI_pairSeq(size)
    XP, XR, Y = preprocess_feature(X, Y, NPZ_PATH["RPI"][size])

if __name__ == "__main__":
    print("Feature Preprocessing")
    
    preprocess_and_savez_RPI(0)