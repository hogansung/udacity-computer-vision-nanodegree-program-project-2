import sys

import torch
import torch.nn as nn
from torchvision import transforms

sys.path.append("/opt/cocoapi/PythonAPI")
from data_loader_wrapper import DataLoaderWrapper
from model import EncoderCNN, DecoderRNN
import math
import nltk

nltk.download("punkt")

## TODO #1: Select appropriate values for the Python variables below.
batch_size = 64
vocab_threshold = 5
vocab_from_file = True
embed_size = 64  # 512
hidden_size = 64  # 512
num_epochs = 100  # number of training epochs
save_every = 1  # determines frequency of saving model weights
print_every = 100  # determines window for printing average loss
log_file = "training_log.txt"  # name of file with saved training loss and perplexity

transform_train = transforms.Compose(
    [
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(224),  # get 224x224 crop from random location
        transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize(
            (0.485, 0.456, 0.406),  # normalize image for pre-trained model
            (0.229, 0.224, 0.225),
        ),
    ]
)

# Build data loader.
data_loader_wrapper = DataLoaderWrapper(
    transform=transform_train,
    batch_size_for_training=batch_size,
    vocab_threshold=vocab_threshold,
    vocab_from_file=vocab_from_file,
    num_workers=4,
)

# The size of the vocabulary.
vocab_size = len(data_loader_wrapper.dataset_for_training.vocab)

# Initialize the encoder and decoder.
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move models to GPU if CUDA is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # overwrite it with cpu
encoder.to(device)
decoder.to(device)

# Define the loss function.
criterion = (
    nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
)
criterion = nn.CrossEntropyLoss()  # overwrite it with cpu

# TODO #3: Specify the learnable parameters of the model.
params = list(encoder.embed.parameters()) + list(decoder.parameters())

# TODO #4: Define the optimizer.
optimizer = torch.optim.Adam(params, lr=0.01)

# Set the total number of training steps per epoch.
total_step = math.ceil(
    len(data_loader_wrapper.dataset_for_training.caption_lengths)
    / data_loader_wrapper.dataset_for_training.batch_size
)


# <a id='step2'></a>
# ## Step 2: Train your Model
#
# Once you have executed the code cell in **Step 1**, the training procedure below should run without issue.
#
# It is completely fine to leave the code cell below as-is without modifications to train your model.  However, if you would like to modify the code used to train the model below, you must ensure that your changes are easily parsed by your reviewer.  In other words, make sure to provide appropriate comments to describe how your code works!
#
# You may find it useful to load saved weights to resume training.  In that case, note the names of the files containing the encoder and decoder weights that you'd like to load (`encoder_file` and `decoder_file`).  Then you can load the weights by using the lines below:
#
# ```python
# # Load pre-trained weights before resuming training.
# encoder.load_state_dict(torch.load(os.path.join('../models', encoder_file)))
# decoder.load_state_dict(torch.load(os.path.join('../models', decoder_file)))
# ```
#
# While trying out parameters, make sure to take extensive notes and record the settings that you used in your various training runs.  In particular, you don't want to encounter a situation where you've trained a model for several hours but can't remember what settings you used :).
#
# ### A Note on Tuning Hyperparameters
#
# To figure out how well your model is doing, you can look at how the training loss and perplexity evolve during training - and for the purposes of this project, you are encouraged to amend the hyperparameters based on this information.
#
# However, this will not tell you if your model is overfitting to the training data, and, unfortunately, overfitting is a problem that is commonly encountered when training image captioning models.
#
# For this project, you need not worry about overfitting. **This project does not have strict requirements regarding the performance of your model**, and you just need to demonstrate that your model has learned **_something_** when you generate captions on the test data.  For now, we strongly encourage you to train your model for the suggested 3 epochs without worrying about performance; then, you should immediately transition to the next notebook in the sequence (**3_Inference.ipynb**) to see how your model performs on the test data.  If your model needs to be changed, you can come back to this notebook, amend hyperparameters (if necessary), and re-train the model.
#
# That said, if you would like to go above and beyond in this project, you can read about some approaches to minimizing overfitting in section 4.3.1 of [this paper](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7505636).  In the next (optional) step of this notebook, we provide some guidance for assessing the performance on the validation dataset.

# In[ ]:


import numpy as np
import os

# Open the training log file.
f = open(log_file, "w")


for epoch in range(1, num_epochs + 1):

    for i_step in range(1, total_step + 1):

        # Obtain the batch.
        images, captions = next(
            iter(data_loader_wrapper.get_data_loader_for_training())
        )

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)
        captions = captions.to(device)

        # Zero the gradients.
        decoder.zero_grad()
        encoder.zero_grad()

        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs = decoder(features, captions)

        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

        # Backward pass.
        loss.backward()

        # Update the parameters in the optimizer.
        optimizer.step()

        # Get training statistics.
        stats = (
            f"Epoch [{epoch}/{num_epochs}], "
            f"Step [{i_step}/{total_step}], "
            f"Loss: {loss.item():.4f}, "
            f"Perplexity: {np.exp(loss.item()):5.4f}"
        )

        # Print training statistics (on same line).
        print("\r" + stats, end="")
        sys.stdout.flush()

        # Print training statistics to file.
        f.write(stats + "\n")
        f.flush()

        # Print training statistics (on different line).
        if i_step % print_every == 0:
            print("\r" + stats)

    # Save the weights.
    if epoch % save_every == 0:
        print(len(encoder.state_dict().keys()))
        torch.save(
            decoder.state_dict(), os.path.join("./models", "decoder-%d.pkl" % epoch)
        )
        torch.save(
            encoder.state_dict(), os.path.join("./models", "encoder-%d.pkl" % epoch)
        )

# Close the training log file.
f.close()


# <a id='step3'></a>
# ## Step 3: (Optional) Validate your Model
#
# To assess potential overfitting, one approach is to assess performance on a validation set.  If you decide to do this **optional** task, you are required to first complete all of the steps in the next notebook in the sequence (**3_Inference.ipynb**); as part of that notebook, you will write and test code (specifically, the `sample` method in the `DecoderRNN` class) that uses your RNN decoder to generate captions.  That code will prove incredibly useful here.
#
# If you decide to validate your model, please do not edit the data loader in **data_loader.py**.  Instead, create a new file named **data_loader_val.py** containing the code for obtaining the data loader for the validation data.  You can access:
# - the validation images at filepath `'/opt/cocoapi/images/train2014/'`, and
# - the validation image caption annotation file at filepath `'/opt/cocoapi/annotations/captions_val2014.json'`.
#
# The suggested approach to validating your model involves creating a json file such as [this one](https://github.com/cocodataset/cocoapi/blob/master/results/captions_val2014_fakecap_results.json) containing your model's predicted captions for the validation images.  Then, you can write your own script or use one that you [find online](https://github.com/tylin/coco-caption) to calculate the BLEU score of your model.  You can read more about the BLEU score, along with other evaluation metrics (such as TEOR and Cider) in section 4.1 of [this paper](https://arxiv.org/pdf/1411.4555.pdf).  For more information about how to use the annotation file, check out the [website](http://cocodataset.org/#download) for the COCO dataset.

# In[ ]:


# (Optional) TODO: Validate your model.
