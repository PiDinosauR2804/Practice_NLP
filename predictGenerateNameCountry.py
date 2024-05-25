from io import open
import glob
import os
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker
hidden_size = 128

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('data/names/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')

# print('# categories:', n_categories, all_categories)
# print(unicodeToAscii("O'Néàl"))

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
     
class RNN1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN1, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input), 1)
        input_combined = self.i2h(input_combined)
        summury = nn.functional.tanh(input_combined + hidden)
        hidden = self.h2h(summury)
        output = self.h2o(summury)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
class RNN_classify(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_classify, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        a =self.i2h(input)
        b = self.h2h(hidden)
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
   
model_classify = RNN_classify(n_letters-1, hidden_size, n_categories) 
model_classify.load_state_dict(torch.load('data/names/model.pth'))

import random

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def inputTensorClassify(line):
    tensor = torch.zeros(len(line), 1, n_letters-1)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# ``LongTensor`` of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)


# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

model = RNN1(input_size=n_letters,hidden_size=hidden_size, output_size=n_letters)

model.load_state_dict(torch.load('data/names/model2.pth'))
criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(category_tensor, input_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = model.initHidden()
    
    model.zero_grad()
    
    loss = torch.Tensor([0])
    for i in range(input_tensor.size(0)):
        output, hidden = model(category_tensor, input_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l
    
    loss.backward()
    
    for p in model.parameters():
        p.data.add(p.grad.data, alpha=learning_rate)
    
    return output, loss.item() / input_tensor.size(0)

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

import matplotlib.pyplot as plt

max_length = 20

n_iters = 100000
print_every = 5000
plot_every = 50
all_losses = []
total_loss = 0 


def evaluate(category_tensor, input_line_tensor):
    hidden = model.initHidden()
    
    for i in range(input_line_tensor.size(0)):
        output, hidden = model(category_tensor, input_line_tensor[0], hidden)
        
    return output

def evaluate_classify(output):
    hidden = model.initHidden()
    inputP = inputTensorClassify(output)
    for i in range(inputP.size(0)):
        output, hidden = model_classify(inputP[i], hidden)
    
    topv, topi = output.topk(2, 1, True)
    predictions = []
    for i in range(2):
        value = topi[0][i].item()
        predictions.append(value)
    return predictions
        
# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = model.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = model(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

# samples('Russian', 'RUS')

# samples('German', 'GER')

# samples('Spanish', 'SPA')

# samples('Chinese', 'CHI')

def eva(ITE):
    wrong = 0
    for i in range(ITE):
        category = randomChoice(all_categories)
        start_letter = randomChoice(all_letters)
        output_name = sample(category=category, start_letter=start_letter)
        # print(output_name)
        pol1, pol2 = evaluate_classify(output_name)
        # print(pol1, pol2)
        country1 = all_categories[pol1]
        country2 = all_categories[pol2]
        if category not in [country1, country2]:
            wrong += 1
        # print("Real: ", category)
        # print("Predict: ",country1, country2)
    print("Result: ", wrong/ITE * 100 ,"%%")
eva(10000)
# print(evaluate_classify('kaka'))
