# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:36:31 2018

@author: Wanyu Du
"""

import collections
import numpy as np

def build_vocab(file_path, freq):
  # get the words and tags
  words=[]
  tags=[]
  with open(file_path) as f:
    lines=f.readlines()
    for line in lines:
      tokens=line.split()
      for token in tokens:
        words.append(token.split('/')[0])
        tags.append(token.split('/')[1])
  tag_cols=list(set(tags))
  tag_cols.sort()
  
  # count the word freqency
  word_counts=collections.Counter(words).most_common()
  idx=0
  vocab={}
  for word, counts in word_counts:
    if counts>freq:
      vocab[word]=idx
      idx+=1
  vocab['UNK']=idx
  return vocab, words, tag_cols


def compute_transition_matrix(file_path, tag_cols):
  # get the tags
  tags_in_line=[]
  with open(file_path) as f:
    lines=f.readlines()
    for line in lines:
      tokens=line.split()
      tags_per_line=[]
      for token in tokens:
        tags_per_line.append(token.split('/')[1])
      tags_in_line.append(tags_per_line)
  
  # compute the transition counts matrix
  cor_matrix=np.zeros((len(tag_cols), len(tag_cols)))
  for tags in tags_in_line:
    for i in range(len(tags)):
      if i==0:
        idx_x=tag_cols.index('START')
        idx_y=tag_cols.index(tags[i])
      elif i==len(tags)-1:
        idx_x=tag_cols.index(tags[i])
        idx_y=tag_cols.index('END')
      else:
        idx_x=tag_cols.index(tags[i])     # y_i_1
        idx_y=tag_cols.index(tags[i+1])   # y_i
      cor_matrix[idx_x][idx_y]+=1
  return cor_matrix


def compute_emission_matrix(file_path, vocab, tag_cols):
  # get the word-tag pairs
  tokens_in_line=[]
  with open(file_path) as f:
    lines=f.readlines()
    for line in lines:
      tokens=line.split()
      tokens_per_line=[]
      for token in tokens:
        items=token.split('/')
        tokens_per_line.append(items)
      tokens_in_line.append(tokens_per_line)
  
  # compute the emission counts matrix
  cor_matrix=np.zeros((len(tag_cols), len(vocab.keys())))
  for line in tokens_in_line:
    for tokens in line:
      idx_x=tag_cols.index(tokens[1])
      if tokens[0] in vocab.keys():
        idx_y=vocab[tokens[0]]
      else:
        idx_y=vocab['UNK']
      cor_matrix[idx_x][idx_y]+=1
  return cor_matrix


def estimate_transition_prob(y_now, y_pre, trans_matrix, tag_cols, beta=0):
  idx_x=tag_cols.index(y_pre)
  idx_y=tag_cols.index(y_now)
  p=(trans_matrix[idx_x][idx_y]+beta)/(np.sum(trans_matrix[idx_x])+len(tag_cols)*beta)
  return p


def estimate_emission_prob(x_now, y_now, emission_matrix, vocab, tag_cols, alpha=0):
  idx_x=tag_cols.index(y_now)
  if x_now in vocab.keys():
    idx_y=vocab[x_now]
  else:
    idx_y=vocab['UNK']
  p=(emission_matrix[idx_x][idx_y]+alpha)/(np.sum(emission_matrix[idx_x])+alpha*len(vocab.keys()))
  return p


def gen_tprob(out_file, trans_matrix, tag_cols, beta):
  outs=open(out_file, 'w', encoding='utf8')
  for tag_i_1 in tag_cols:
    for tag_i in tag_cols:
      p=estimate_transition_prob(y_now=tag_i, y_pre=tag_i_1, trans_matrix=trans_matrix,
                                 tag_cols=tag_cols, beta=beta)
      outs.write(tag_i_1+','+tag_i+','+str(p)+'\n')
  outs.close()


def gen_eprob(out_file, emission_matrix, words, tag_cols, vocab, alpha):
  outs=open(out_file, 'w', encoding='utf8')
  for tag in tag_cols:
    for word in words:
      if tag=='START' or tag=='END':
        continue
      else:
        p=estimate_emission_prob(x_now=word, y_now=tag, emission_matrix=emission_matrix,
                                 vocab=vocab, tag_cols=tag_cols, alpha=alpha)
        outs.write(tag+','+word+','+str(p)+'\n')
  outs.close()

  
def viterbi(sentence, vocab, tag_cols, trans_matrix, emission_matrix, alpha, beta):
  v=np.zeros((len(sentence)+1, len(tag_cols)-2))
  b=np.zeros((len(sentence)+1, len(tag_cols)-2))
  s=np.zeros((1, len(tag_cols)-2))
  # calculate s(y0, START), v(x0)
  for k in range(1, len(tag_cols)-1):
    tp=estimate_transition_prob(y_now=tag_cols[k], y_pre='START', trans_matrix=trans_matrix, 
                                tag_cols=tag_cols, beta=beta)
    ep=estimate_emission_prob(x_now=sentence[0], y_now=tag_cols[k], emission_matrix=emission_matrix, 
                              vocab=vocab, tag_cols=tag_cols, alpha=alpha)
    v[0][k-1]=np.log(tp)+np.log(ep)
    b[0][k-1]=tag_cols.index('START')
  
  # calculate s(yi, yi_1), v(xi)
  for m in range(1, len(sentence)):
    for k in range(1, len(tag_cols)-1):
      for kk in range(1, len(tag_cols)-1):
        tp=estimate_transition_prob(y_now=tag_cols[k], y_pre=tag_cols[kk], trans_matrix=trans_matrix,
                                    tag_cols=tag_cols, beta=beta)
        ep=estimate_emission_prob(x_now=sentence[m], y_now=tag_cols[k], emission_matrix=emission_matrix, 
                                  vocab=vocab, tag_cols=tag_cols, alpha=alpha)
        s[0][kk-1]=np.log(tp)+np.log(ep)
      v[m][k-1]=np.max(v[m-1]+s[0])
      b[m][k-1]=np.argmax(v[m-1]+s[0])+1    # plus 1 to align with the index in tag_cols
    
  # calculate s(END, yi), v(END)
  for k in range(1, len(tag_cols)-1):
    for kk in range(1, len(tag_cols)-1):
      tp=estimate_transition_prob(y_now='END', y_pre=tag_cols[kk], trans_matrix=trans_matrix, 
                                  tag_cols=tag_cols, beta=beta)
      ep=1
      s[0][kk-1]=np.log(tp)+np.log(ep)
    v[len(sentence)][k-1]=np.max(v[len(sentence)-1]+s[0])
    b[len(sentence)][k-1]=np.argmax(v[len(sentence)-1]+s[0])+1
    
  # get the predict tags
  m_idx=np.array(np.arange(1, len(sentence)))
  m_idx=m_idx[::-1]
  y_m=[]
  y_m.append(tag_cols[int(b[len(sentence)][0])])
  for i, m in enumerate(m_idx):
    b_last=tag_cols.index(y_m[i])
    b_now=b[m][int(b_last)-1]
    y_m.append(tag_cols[int(b_now)])
  y_m.reverse() 
  return y_m, v, b


def get_dev_acc(alpha, beta):
  acc_num=0.
  total_num=0.
  with open('dev.pos') as f:
    lines=f.readlines()
    for line in lines:
      tokens=line.split()
      sent=[]
      label=[]
      for token in tokens:
        items=token.split('/')
        sent.append(items[0])
        label.append(items[1])
      preds, v, b=viterbi(sentence=sent, vocab=vocab, tag_cols=tag_cols, 
                          trans_matrix=trans_matrix, emission_matrix=emission_matrix, 
                          alpha=alpha, beta=beta)
      for pred, truth in zip(preds, label):
        if pred==truth:
          acc_num+=1
      total_num+=len(label)
  return acc_num/total_num

  
if __name__=='__main__':
  # build vocab
  vocab, words, tag_cols=build_vocab(file_path='trn.pos', freq=3)
  tag_cols.insert(0, 'START')
  tag_cols.insert(len(tag_cols), 'END')
  
  # get transition table
  trans_matrix=compute_transition_matrix(file_path='trn.pos', tag_cols=tag_cols)
  # get emission table
  emission_matrix=compute_emission_matrix(file_path='trn.pos', vocab=vocab, tag_cols=tag_cols)
  
  # predict the word-tag pairs
  a=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  b=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  for aa in a:
    for bb in b:
      acc=get_dev_acc(aa, bb)
      print('alpha='+str(aa)+', beta='+str(bb)+', overal_accuracy=', acc)
  