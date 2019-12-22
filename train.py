import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from policy_gradient import PolicyGradient

parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', help='use GPU')


class Params:
    NUM_EPOCHS = 50
    ALPHA = 5e-3        # learning rate
    BATCH_SIZE = 3     # how many episodes we want to pack into an epoch
    HIDDEN_SIZE = 64    # number of hidden nodes we have in our dnn
    BETA = 0.1          # the entropy bonus multiplier
    INPUT_SIZE = 3
    ACTION_SPACE = 3
    NUM_STEPS = 4
    GAMMA = 0.99


def main():
    args = parser.parse_args()
    use_cuda = args.use_cuda
    use_cuda = True

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)


    policy_gradient = PolicyGradient(config=Params, train_set=trainloader, test_set=testloader, use_cuda=use_cuda)
    policy_gradient.solve_environment()


if __name__ == "__main__":
    main()