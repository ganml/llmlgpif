# llama with ollama
import math
from datetime import datetime
import psutil
import pandas as pd
from langchain_ollama import OllamaLLM


tr = pd.read_csv('peril.training.csv')
te = pd.read_csv('peril.validation.csv')

descriptions_tr = set([i.rstrip() for i in list(tr['Description'])])
descriptions_te = set([i.rstrip() for i in list(te['Description'])])
descriptions = list(descriptions_tr.union(descriptions_te))
print(len(descriptions))

checkpoint0 = datetime.now() # start runtime counter

# ollama model ids
model_id = "llama3.2" # llama3.2 3B - default
#model_id = "llama3.2:1b" # llama3.2 1B

# initialize the model
ollama_server_url = "http://localhost:11434"
llm = OllamaLLM(model = model_id, base_url = ollama_server_url, num_thread=10)

checkpoint1 = datetime.now() # runtime counter

# prompts 
#categories = ['low', 'high']
#categories = ['low', 'medium', 'high']
categories = ['negligible', 'low', 'medium', 'high', 'catastrophic']
levels = len(categories)
instruction = "You are an actuarial assistant assigned to classify these insurance incidents into one of " + str(levels) + " categories: "
for i in range(levels-1):
    instruction += categories[i] + ', '
instruction += categories[levels-1]

N = len(descriptions)
batch_size = 20
batches = N // batch_size

labels = [] 
for i in range(batches+1):
    beg = i * batch_size
    end = min(N, beg + batch_size)
    print(beg)
    if beg==end:
        break
    prompts = 'The following are descriptions of ' + str(end-beg) + ' insurance incidents:'
    for j in range(beg, end):
        prompts += '\n' + str(j-beg+1) + '. ' + descriptions[j]
    prompts += '\n' + instruction
    prompts += f'. Return only {end-beg} lines. Each line contains only the classification of the corresponding incident. Except for the classification name, do not include other information in the output. Format the output as:\n1. classification category\n2. classification category\n...'
    input = [{"role": "user", "content": prompts }]
    response = llm.invoke(input)
    lines = response.split('\n')
    print(response)
    labels.extend([item for item in lines if len(item)>0 and item[0].isdigit()])
    
#print(labels)
#quit()

checkpoint2 = datetime.now() # end runtime counter
print(len(labels))

def extractLabel(line):
    ind = line.index('.')
    return line[ind+1:].strip()

label_tr = [] 
for i in tr.index: 
    val = tr['Description'][i].rstrip()
    ind = descriptions.index(val)
    if ind >= 0:
        label_tr.append(extractLabel(labels[ind]))
    else:
        print(val)
tr['label'] = label_tr

label_te = [] 
for i in te.index: 
    val = te['Description'][i].rstrip()
    ind = descriptions.index(val)
    if ind >= 0:
        label_te.append(extractLabel(labels[ind]))
    else:
        print(val)
te['label'] = label_te

# get runtime and cpu, ram usage:
total_runtime = checkpoint2 - checkpoint0
request_runtime = checkpoint2 - checkpoint1
# compile into dataframe
output = {"model": [model_id], "n cases": [len(descriptions_tr)+len(descriptions_te)], "total runtime (min)": [total_runtime/60], "request": [request_runtime/60]}
output = pd.DataFrame(output)

# create an output file in the parent directory
tr.to_csv("n" + str(levels) +'peril.training.csv', index=None)
te.to_csv("n" + str(levels) +'peril.validation.csv', index=None)
output.to_csv("n" + str(levels) + ".csv", index = None)
