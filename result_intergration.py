import os
import sys
import glob
from multiprocessing.dummy import Pool as ThreadPool
from pickletools import float8
import pandas as pd
from Bio import SeqIO
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

def seq_fasta(seq_file,record):
    Seqs = SeqIO.parse(seq_file,'fasta')
    for seq in Seqs:
        record['seq'].append(seq.seq)
    return record
def r_lstm(result_lstm,record):
    with open(result_lstm,'r') as f:
        for score in f:
            score=score.rstrip()
            if float(score)>0.5:
                record['lstm'].append(1)
            else:
                record['lstm'].append(0)
    return record
def r_bert(result_bert,record):
    with open(result_bert,'r') as f:
        for score in f:
            score=score.rstrip()
            if float(score)>0.5:
                record['bert'].append(1)
            else:
                record['bert'].append(0)
    return record

def r_att(result_trans,record):
    with open(result_trans,'r') as f:
        for score in f:
            score=score.rstrip()
            if float(score)>0.5:
                record['trans'].append(1)
            else:
                record['trans'].append(0)
    return record

def get_filePath(file_path):
    path_ls = []
    print(file_path)
    for path in glob.glob(file_path +'*'):
        if path.split('.')[2] == 'fa':
            path_ls.append(path)
    return path_ls

def data_process(path):
    seq_file     = path
    result_lstm  = '.' + path.split('.')[1] + '_lstm.txt'
    result_bert  = '.' + path.split('.')[1] + '_bert.txt'
    result_trans = '.' + path.split('.')[1] + '_att.txt'
    output_csv   = '.' + path.split('.')[1] + '_positive.csv'
    record={}
    record['seq']   = []
    record['lstm']  = []
    record['bert']  = []
    record['trans'] = []
    record = seq_fasta(seq_file,record)
    record =r_lstm(result_lstm,record)
    record =r_bert(result_bert,record)
    record = r_att(result_trans,record)
    record=pd.DataFrame(record)
    record["total"] =record[['lstm','bert','trans']].parallel_apply(lambda x:x.sum(),axis =1)
    print('=================finish sum operation====================')
#record.total = record.sum(axis=0)
    record['prediction'] = record["total"].parallel_apply(lambda x :1 if x==3 else 0)
    print('=================finish prediction=======================')
#record['prediction'] = [1 if total==3 else 0 for total in record["total"]]
    record = record[record.prediction==1]
    record =record['seq']
    record.to_csv(output_csv,header=0,index=0)
    print('=================finish writing format of csv============')

#record.index = record['seq']
#record = record.drop(['seq'], axis=1)
#record.columns=["lstm","bert","trans"]
#record.total = record.sum(axis=0)
#record['prediction'] = [1 if total==3 else 0 for total in record["total"]]

if __name__=='__main__':
    file_path = './final_result_1/'
    path_ls = get_filePath(file_path)
    for path in path_ls:
        print(path)
        data_process(path)
'''
rec =[]
fd =open(output_fasta,'w')
for seq in SeqIO.parse(seq_file,'fasta'):
    if seq.seq in set(record['seq']):
        rec.append(seq)
SeqIO.write(rec,fd,'fasta')
fd.close()
'''
