import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        assert features.shape[0] == captions.shape[0], "Batch sizes are differrent for features and captions."
        device = features.get_device()

        # ignore the last caption, which is <END>
        inputs = torch.cat((features.view(-1, 1, self.embed_size), self.word_embeddings(captions[:,:-1])), dim=1)
#         print(inputs.shape)

        h0 = torch.randn(self.num_layers, inputs.shape[1], self.hidden_size).to(device)
        c0 = torch.randn(self.num_layers, inputs.shape[1], self.hidden_size).to(device)
        outputs, hidden = self.lstm(inputs, (h0, c0))
#         print(outputs.shape)

        return self.fc(outputs)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass