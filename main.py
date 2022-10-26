#This program is aim at construct a simple hybrid
#GAN with the method of WGAN-GP

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
from utils import gradient_penalty, transmission_by_specturm

#Hyperpremeters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 128
CHANNEL_IMG = 1
Z_DIM = 100
NUM_EPOCH = 20
FEATURES_critic = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

#Optical parameter
pi = np.pi
lam = 632.8*1e-9
pix = 12.5*1e-6
M = 128
all_size = pix*M

#Discriminator's parameters
layer_interval = 20*1e-3
in_distance = 30*1e-3
out_distance = 30*1e-3

#Prepare dataset
transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Pad(int((M - IMAGE_SIZE) / 2)),
        transforms.Normalize(
            [0.5 for _ in range(CHANNEL_IMG)], [0.5 for _ in range(CHANNEL_IMG)]
        ),
    ]
)

#train data
train_dataset = torchvision.datasets.MNIST(root="./dataset", train=True,
                                        transform=transforms, download=True)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

#test data
test_dataset = torchvision.datasets.MNIST(root="./dataset", train=False,
                                        transform=transforms, download=True)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)


#Instance Networks
critic = Discriminator(CHANNEL_IMG, FEATURES_critic).to(device)
initialize_weights(critic)

gen = Generator(all_size, in_distance, layer_interval, out_distance, lam, M, M).to(device)
initialize_weights(gen)

#Optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

#Criterion
# criterion = nn.CrossEntropyLoss()


fixed_noise = torch.exp(1j*0.2*torch.randn(32, 1, M, M)).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
writer_out = SummaryWriter(f"logs/out")
writer_accuracy = SummaryWriter(f"logs/accuracy")
step = 0

gen.train()
critic.train()

#Train
for epoch in range(NUM_EPOCH):
    for batch_idx, data in enumerate(train_loader, 0):
        real, targets = data
        real, targets = real.to(device), targets.to(device)

        #input noise
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.exp(1j*0.2*torch.randn(BATCH_SIZE, 1, M, M)).to(device)
            # tag = torch.ones((BATCH_SIZE, 50, 1, 1), device=device)*targets.reshape((BATCH_SIZE, 1, 1, 1))
            # noise_tag = torch.cat([noise, tag], dim=1)
            # if (epoch==0 and batch_idx==0):
            #     fixed_noise = noise_tag
            fake = gen(noise).to(torch.float32)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            # critic_real = torch.sum(critic_real_r, dim=1).view(-1)
            # critic_fake = torch.sum(critic_fake_r, dim=1).view(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp)
            # loss_critic2_1 = criterion(critic_real_r, targets)
            # loss_critic2_2 = criterion(critic_fake_r, targets)
            # loss_critic2 = loss_critic2_1 + loss_critic2_2
            loss_critic = loss_critic
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        ### Train Generater: min -E[critic(gen_fake)]
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasinonally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch[{epoch}/{NUM_EPOCH}] Batch {batch_idx}/{len(train_loader)} \ "
                f"WLoss D: {loss_critic:.4f}, Wloss G: {loss_gen:.4f} \ "
                # f"Loss Real: {loss_critic2_1:.4f}, loss Fake: {loss_critic2_2:.4f}"

            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1

    #
    # #test
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in test_loader:
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = critic(images)
    #         _, predicted = torch.max(outputs.data, dim=1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # print('Accuracy on test set: %d %%' % (100 * correct/total))
    #
    # writer_accuracy.add_scalar("Accuracy", (100 * correct/total), global_step=epoch)


