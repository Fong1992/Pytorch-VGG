import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import datasets, transforms

VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def Params_Num(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} Total params.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} Trainable params.')


def Load_Data(data_dir):
    data_transforms = {x: transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomRotation(degrees=15),
         transforms.ColorJitter(),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        for x in {'train', 'val'}}

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    return {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=70, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}


class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.features = self.create_conv_layers(VGG_types['VGG16'])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


## vgg16 model with no weight
# result = model(x)
# print(torch.argmax(result), result[0, torch.argmax(model(x)).item()])

## torchvision vgg16
# import torchvision.models as models
# vgg16 = models.vgg16(pretrained=True).to(device)
# vgg16 = vgg16.eval()
# result = vgg16(x)
# print(torch.argmax(result), result[0, torch.argmax(model(x)).item()])


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_dir = 'D:/Data/DataSet/dog_vs_cat/'
    dataloaders = Load_Data(data_dir)

    model = VGG_net(in_channels=3, num_classes=1000)
    ## vgg16 with pre-train weight
    model.load_state_dict(torch.load('vgg16.pth'))

    for parma in model.parameters():
        parma.requires_grad = False

    # model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    # model.classifier[-1] = nn.Linear(4096, 2)

    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2))


    model = model.to(device)
    Params_Num(model)

    num_epochs = 10
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print('Epoch: {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 40)
        for phase in {'train', 'val'}:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for train_x, train_y in dataloaders[phase]:
                train_x = train_x.to(device)
                train_y = train_y.to(device)

                optimizer.zero_grad()

                pred_y = model(train_x)
                _, pred_ind = torch.max(pred_y, 1)

                loss = criterion(pred_y, train_y)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(pred_ind == train_y.data).item()

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = 100 * running_corrects / (len(dataloaders[phase]) * dataloaders[phase].batch_size)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
