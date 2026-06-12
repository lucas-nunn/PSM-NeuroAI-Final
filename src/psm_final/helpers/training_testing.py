# CREDIT: ADRIEN DOERIG

import torch
from tqdm import tqdm

def train(model, loader, n_epochs, criterion, optimizer, device):
    '''
    model: a pytorch model instance
    loader: a pytorch DataLoader instance
    n_epochs: number of epochs to train for
    criterion: a pytorch loss function instance
    optimizer: a pytorch optimizer instance
    device: 'cuda' or 'cpu'
    '''

    # Set the model to training mode. This helps inform layers such as Dropout
    # and BatchNorm, which are designed to behave differently during training
    # and evaluation. For instance, in training mode, dropout "kills" neurons
    # in each batch, but this is not usually done while testing.
    # In our case, we are not using any of these fancy things, but it is good to
    # get into the habit.
    model.train(True)

    history = []  # we will record the losses over training here.

    # We loop over the whole dataset n_epochs times
    for epoch in range(n_epochs):

        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        # The tqdm python module allows us to have a nice progress bar during
        # training. Don't be confused if it looks complicated.
        # It is just a fancy for loop. We loop over the dataloader and
        # get each item=(batch_imgs, batch_labels) each time.
        for i, item in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}")):

            optimizer.zero_grad()  # Set all gradients to 0

            inputs, labels = item[0], item[1]  # Pair of input and corresponding label
            inputs = inputs.to(device)         # Move to desired device (GPU/CPU)
            labels = labels.to(device)         # Move to desired device (GPU/CPU)

            outputs = model(inputs)            # Feed the input through the model
            batch_loss = criterion(outputs, labels)  # Calculate the loss

            batch_loss.backward()  # Calculate the gradient for the current loss
            optimizer.step()       # Update weights via backpropagation

            # record loss for this batch as np value. Note that we need all the
            # complicated .cpu().detach().numpy() because the loss is a torch
            # tensor, which includes all the information required to do
            # backprop, use the GPU, etc. We just want normal old numpy.
            history.append(batch_loss.cpu().detach().numpy())

            # Print the current batch loss every few batches
            if i%5==0:
              tqdm.write(f"\rBatch Loss: {batch_loss.item()}", end='')

    return history

def test(model, loader, criterion, device):
    '''
    model: a pytorch model instance
    loader: a pytorch DataLoader instance
    criterion: a pytorch loss function instance
    device: 'cuda' or 'cpu'
    '''

    # Set the model to evaluation mode. Again, this is important when using
    # more complex ingredients, such as BatchNorm or Dropout, which we are not
    # using here. Still, it is good to get into the habit.
    model.eval()

    # initialize the values we'll report
    test_loss = 0 # this is for the cross-entropy loss
    correct = 0 # we'll get a %correct accuracy
    total = 0 # this is to count the total number of test examples seen.

    # Disable gradient calculation (i.e., we don't keep track of which units
    # impact the loss as we don't want to learn with gradient descent during
    # testing).
    with torch.no_grad():

        # Loop over the testing dataloader, with a nice tqdm progressbar
        for item in tqdm(loader):

            inputs, labels = item[0], item[1]
            inputs = inputs.to(device)
            labels = labels.to(device)

            # pass data through model
            outputs = model(inputs)

            # add loss to total test loss
            test_loss += criterion(outputs, labels).item() # note: a += b is the same as a = a+b

            # to compute the accuracy, we need to get the output with the
            # highest value (this is the network's prediction)
            _, predicted = torch.max(outputs.data, 1)
            # this is a logical operation to get the number of items
            # where the prediction matches the label
            correct += (predicted == labels).sum().item()

            # keep track of how many test samples we've seen
            total += labels.size(0)

    test_loss /= len(loader) # report average loss per batch
    accuracy = 100 * correct / total # report percent correct
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return test_loss, accuracy

def train_and_test(model, dataloader_train, dataloader_test, n_epochs, criterion, optimizer, device):
    '''
    model: a pytorch model instance
    dataloader_train: a pytorch DataLoader instance for training
    dataloader_test: a pytorch DataLoader instance for testing
    n_epochs: number of epochs to train for
    criterion: a pytorch loss function instance
    optimizer: a pytorch optimizer instance
    device: 'cuda' or 'cpu'
    '''

    history = []

    # We loop over the whole training dataset n_epochs times
    for epoch in range(n_epochs):

        print(f"Epoch {epoch + 1}/{n_epochs}")

        # Set the model to training mode.
        model.train(True)

        # Do training loop.
        # THIS IS ESSENTIALLY THE SAME AS OUR PREVIOUS TRAINING FUNCTION
        for i, item in enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch + 1}")):
            optimizer.zero_grad()  # Set all gradients to 0
            inputs, labels = item[0], item[1]  # Pair of input and corresponding label
            inputs = inputs.to(device)         # Move to desired device (GPU/CPU)
            labels = labels.to(device)         # Move to desired device (GPU/CPU)
            outputs = model(inputs)            # Feed the input through the model
            batch_loss = criterion(outputs, labels)  # Calculate the loss
            history.append(batch_loss.cpu().detach().numpy())
            batch_loss.backward()  # Calculate the gradient for the current loss
            optimizer.step()
            # Print the current batch loss every few batches
            if i%5==0:
              tqdm.write(f"\rBatch Loss: {batch_loss.item()}", end='')

        # After each epoch, evaluate the model on the test set
        test_loss, accuracy = test(model, dataloader_test, criterion, device)

    return history