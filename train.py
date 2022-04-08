from shutil import which
from load import BirdClefDataset, MusicDataset, BirdClefWavDataset


import torch
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from models import BirdConv1d, BirdConv2d
from timeit import default_timer as timer


def run(which_model, which_data):
    print("Using model", which_model)
    print("Using data", which_data)
    model_name = which_data + "_" + which_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    batch_size =64
    n_samples = 64000

    if device == "cuda":
        num_workers = 10
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False



    # if which_model == "1d":
    #    sample_rate=32000
    #    new_sample_rate = 8000
    #     transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    # elif which_model == "2d":
    #     transform = torchaudio.transforms.Spectrogram(n_fft=400, hop_length=50)


    if which_data == "birds":
        full_dataset = BirdClefDataset(n_samples=n_samples)
    elif which_data == "birds-wav":
        full_dataset = BirdClefWavDataset(n_samples=n_samples)
    elif which_data == "music":
        full_dataset = MusicDataset(n_samples=n_samples)

    print("labels", full_dataset.labels)

    #Use N total examples 
    tot_size = 10000
    extra_size = len(full_dataset) - tot_size
    subset, _ = torch.utils.data.random_split(full_dataset, [tot_size, extra_size])


    train_size = int(0.90 * len(subset))
    test_size = len(subset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(subset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )




    if which_model == "1d":
        model = BirdConv1d(n_input=1, n_output=len(full_dataset.labels))
    elif which_model == "2d":
        model = BirdConv2d(n_input=1, n_output=len(full_dataset.labels))



    model.to(device)
    print(model)
    n = count_parameters(model)
    print("Number of parameters: %s" % n)


    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    log_interval = 10
    n_epoch = 2000


    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval, train_loader, device, optimizer, model_name)
        test(model, epoch, train_loader, test_loader, device)
        scheduler.step()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def train(model, epoch, log_interval, train_loader, device, optimizer, model_name):
    model.train()
    start_time = timer()
    for batch_idx, (data, target) in enumerate(train_loader):

        
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        #data = transform(data)
        
        output = model(data)



        loss = F.binary_cross_entropy_with_logits(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time = timer()
        # print training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tTime: {end_time - start_time:.2f}")
            torch.save(model.state_dict(), 'models/' + model_name)
        start_time = timer()

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def test(model, epoch, train_loader, test_loader, device):
    model.eval()
    correct = 0

    data, target = next(iter(train_loader))
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    pred = get_likely_index(output)
    correct = number_of_correct(pred, target)
    print("train predicted: ", pred.squeeze()[:15])
    print("train target:    ", target[:15])
    print(f"train pct:({100. * correct / len(target):.0f}%)")
    correct = 0

    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)
        n_labels = list(target.size())[-1]


        output = model(data)

        #output = F.softmax(output.squeeze(), dim=1) 

        pred = get_likely_index(output)
        target = get_likely_index(target)
        correct += number_of_correct(pred, target)

    print(f"\nTest Epoch: {epoch}")
    print(f"\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * (correct) / (len(test_loader.dataset)):.0f}%)")

    print(f"\ttotal examples: {len(test_loader.dataset)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="1d", help="model: 1d or 2d")
    parser.add_argument("--data", type=str, default="birds", help="data source: birds or music")
    args = parser.parse_args()
    which_model = args.model
    which_data = args.data
    run(which_model, which_data)