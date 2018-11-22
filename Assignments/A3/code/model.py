import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ShallowNet(nn.Module):
    # in this class, a frozen ResNet50 architecture is used to obtain high-level features to shallow architecture.
    def __init__(self, transfer_learning=False):
        super(ShallowNet, self).__init__()
        self.transfer_learning = transfer_learning
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        if transfer_learning:
            self.fc0 = nn.Linear(832, 100)
            self.fc1 = nn.Linear(100, 50)
        else:
            self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

        if transfer_learning:
            resnet_model = torchvision.models.resnet18(pretrained=True)
            for param in resnet_model.parameters():
                param.requires_grad = False
            # remove last layer
            self.resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))


    def forward(self, x):
        standard_x, resnet_x = x
        if self.transfer_learning:
            resnet_x = self.resnet_model(resnet_x)
            resnet_x = resnet_x.view(-1, 512)
        standard_x = F.relu(F.max_pool2d(self.conv1(standard_x), 2))
        standard_x = F.relu(F.max_pool2d(self.conv2(standard_x), 2))
        standard_x = F.relu(F.max_pool2d(self.conv3(standard_x), 2))
        standard_x = standard_x.view(-1, 320)

        if self.transfer_learning:
            x = torch.cat((standard_x, resnet_x), 1)
            x = F.relu(self.fc0(x))
        else:
            x = standard_x

        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PretrainedNet(nn.Module):
    # In this model, we fit a ResNet 50 to the data by frozening the first layers and unfrozening the last ones
    def __init__(self, model_name='inception', n_frozen_layers=14):
        super(PretrainedNet, self).__init__()
        self.fc2 = nn.Linear(50, nclasses)

        if model_name == 'resnet18':
            self.pretrained_model = torchvision.models.resnet18(pretrained=True)
        elif model_name == 'resnet50':
            self.pretrained_model = torchvision.models.resnet50(pretrained=True)
        elif model_name =='inception':
            self.pretrained_model = torchvision.models.inception_v3()
        else:
            raise ValueError("Pretrained model '{}' is not supported".format(model_name))

        # froze first layers
        count = 0
        for name, child in self.pretrained_model.named_children():
            count += 1
            if count <= n_frozen_layers:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
        # replace the last layer with a layer with the right output dimension
        num_ftrs = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(num_ftrs, nclasses)


    def forward(self, x):
        _, x = x
        return self.pretrained_model(x)


