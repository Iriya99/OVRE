import torch
import pickle
from tqdm import tqdm
import torch.nn.functional as nnf
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from scipy.optimize import linear_sum_assignment
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").cuda()

bleu = Bleu(4)
meteor = Meteor()
rouge = Rouge()
cider = Cider()

refs = {}
hypos = {}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--result_path", default="valor10.txt")
args = parser.parse_args()

with open("/home/wtt/results_/results_test.txt", "r") as f:
    gts = f.readlines()

with open("/home/wtt/results/" + args.result_path, "r") as f:
    gens = f.readlines()

sep = " / "

for i in range(len(gts)):
    tmp = []
    for gt in gts[i].split("||")[1].split(sep):
        gt = gt.strip("\n")
        if gt not in tmp:
            tmp.append(gt)
    refs[int(gts[i].split("||")[0])] = tmp

for i in range(len(gens)):
    tmp = []
    for gen in gens[i].split("||")[1].split(sep):
        gen = gen.strip("\n")
        if gen not in tmp:
            tmp.append(gen)
    hypos[int(gens[i].split("||")[0])] = tmp

metric_B1 = 0.
metric_B2 = 0.
metric_B3 = 0.
metric_M = 0.
metric_R = 0.
metric_C = 0.
metric_Sim = 0.
num_refs = list()
num_min =list()
ix2refs = dict()
ix2hypos = dict()

bar = tqdm(sorted(refs.keys()))
dic_ix = 0

S = []
for i in bar:

    refs_token = tokenizer(refs[i], padding=True, truncation=True, return_tensors="pt")
    hypos_token = tokenizer(hypos[i], padding=True, truncation=True, return_tensors="pt")

    num_refs.append(len(refs[i]))
    num_min.append(min(len(refs[i]), len(hypos[i])))

    for k, v in refs_token.items():
        refs_token[k] = v.cuda()
    for k, v in hypos_token.items():
        hypos_token[k] = v.cuda()

    with torch.no_grad():
        refs_embeddings = model(**refs_token, output_hidden_states=True, return_dict=True).pooler_output
        hypos_embeddings = model(**hypos_token, output_hidden_states=True, return_dict=True).pooler_output

    sim_matrix = torch.zeros(len(refs[i]), len(hypos[i]))
   
    for j in range(len(refs[i])):
        sim_matrix[j] = nnf.cosine_similarity(refs_embeddings[j], hypos_embeddings)

    refs_idx, hypos_idx = linear_sum_assignment(1-sim_matrix)
    
    for j in range(len(refs_idx)):
        ix2refs[dic_ix] = [refs[i][refs_idx[j]]]
        ix2hypos[dic_ix] = [hypos[i][hypos_idx[j]]]
        S.append(sim_matrix[refs_idx[j], hypos_idx[j]])
        dic_ix += 1

B = torch.tensor(bleu.compute_score(ix2refs, ix2hypos)[1])
M = torch.tensor(meteor.compute_score(ix2refs, ix2hypos)[1])
R = torch.tensor(rouge.compute_score(ix2refs, ix2hypos)[1])
C = torch.tensor(cider.compute_score(ix2refs, ix2hypos)[1])
S = torch.stack(S)

assert B.shape[1] == M.shape[0] == R.shape[0] == C.shape[0] == S.shape[0]

now_ix = 0
bar = tqdm(range(len(num_min)))
for i in bar:
    metric_B1 += B[0][now_ix:now_ix+num_min[i]].sum() / num_refs[i]
    metric_B2 += B[1][now_ix:now_ix+num_min[i]].sum() / num_refs[i]
    metric_B3 += B[2][now_ix:now_ix+num_min[i]].sum() / num_refs[i]
    metric_M += M[now_ix:now_ix+num_min[i]].sum() / num_refs[i]
    metric_R += R[now_ix:now_ix+num_min[i]].sum() / num_refs[i]
    metric_C += C[now_ix:now_ix+num_min[i]].sum() / num_refs[i]
    metric_Sim += S[now_ix:now_ix+num_min[i]].sum() / num_refs[i]
    now_ix += num_min[i]
    bar.set_postfix(B1=metric_B1/(i+1)*100, 
                    B2=metric_B2/(i+1)*100, 
                    B3=metric_B3/(i+1)*100,
                    meteor=metric_M/(i+1)*100, 
                    rouge=metric_R/(i+1)*100, 
                    cider=metric_C/(i+1)*100, 
                    sim=metric_Sim/(i+1)*100)
