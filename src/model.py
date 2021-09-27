import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
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
            input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        assert (
            features.shape[0] == captions.shape[0]
        ), "Batch sizes are different for features and captions."

        embeddings = self.word_embeddings(captions[:, :-1])
        list_of_inputs = torch.cat((features, embeddings), dim=1)
        list_of_outputs, _ = self.lstm(list_of_inputs, None)
        return self.fc(list_of_outputs)

    def sample(self, features, hidden=None, max_len=20):
        """
        accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)
        """
        inputs = features

        token_ids = []
        for _ in range(max_len):
            outputs, hidden = self.lstm(inputs, hidden)
            outputs = self.fc(outputs)
            _, token_id = outputs.max(2)
            token_ids.append(token_id.item())
            inputs = self.word_embeddings(token_id)

        return token_ids
