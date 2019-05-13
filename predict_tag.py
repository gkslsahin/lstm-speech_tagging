# import resources
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from .lstm_tagger import LSTMTagger


# training sentences and their corresponding word-tags
training_data = [
    ("The cat ate the cheese".lower().split(), ["DET", "NN", "V", "DET", "NN"]),
    ("She read that book".lower().split(), ["NN", "V", "DET", "NN"]),
    ("The dog loves art".lower().split(), ["DET", "NN", "V", "NN"]),
    ("The elephant answers the phone".lower().split(), ["DET", "NN", "V", "DET", "NN"])
]

#dictionary that maps words to indices
word2idx = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

#dictionary that maps tags to indices
tag2idx = {"DET": 0, "NN": 1, "V": 2}



# a helper function for converting a sequence of words to a Tensor of numerical values
def prepare_sequence(seq, to_idx):
    '''This function takes in a sequence of words and returns a 
    corresponding Tensor of numerical values (indices for each word).'''
    idxs = [to_idx[w] for w in seq]
    idxs = np.array(idxs)
    return torch.from_numpy(idxs)

example_input = prepare_sequence("The dog answers the phone".lower().split(), word2idx)
print(example_input)


EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# instantiate our model
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2idx), len(tag2idx))

#loss and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


#test before train

test_sentence = "The cheese loves the elephant".lower().split()

inputs = prepare_sequence(test_sentence, word2idx)
inputs = inputs
tag_scores = model(inputs)
print(tag_scores)

# recall that these numbers correspond to tag2idx = {"DET": 0, "NN": 1, "V": 2}
_, predicted_tags = torch.max(tag_scores, 1)
print('\n')
print('Predicted tags: \n',predicted_tags)

n_epochs = 300

for epoch in range(n_epochs):
    
    epoch_loss = 0.0
    
    # get all sentences and corresponding tags in the training data
    for sentence, tags in training_data:
        
        # zero the gradients
        model.zero_grad()

        # zero the hidden state of the LSTM, this detaches it from its history
        model.hidden = model.init_hidden()

        # prepare the inputs for processing by out network, 
        # turn all sentences and targets into Tensors of numerical indices
        sentence_in = prepare_sequence(sentence, word2idx)
        targets = prepare_sequence(tags, tag2idx)

        # forward pass to get tag scores
        tag_scores = model(sentence_in)

        # compute the loss, and gradients 
        loss = loss_function(tag_scores, targets)
        epoch_loss += loss.item()
        loss.backward()
        
        # update the model parameters with optimizer.step()
        optimizer.step()
        
    # print out avg loss per 20 epochs
    if(epoch%20 == 19):
        print("Epoch: %d, loss: %1.5f" % (epoch+1, epoch_loss/len(training_data)))


#test after the training
test_sentence = "The cheese loves the elephant".lower().split()

# see what the scores are after training
inputs = prepare_sequence(test_sentence, word2idx)
inputs = inputs
tag_scores = model(inputs)
print(tag_scores)

# print the most likely tag index, by grabbing the index with the maximum score!
# recall that these numbers correspond to tag2idx = {"DET": 0, "NN": 1, "V": 2}
_, predicted_tags = torch.max(tag_scores, 1)
print('\n')
print('Predicted tags: \n',predicted_tags)