import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import model_dict
from utils import choose_dataset, validate

# Define the BasicBlock for ResNet18
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Define ResNet18 model using BasicBlock
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)


        pre_pool=out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        pre_out=out
        out = self.fc(out)
        return out, pre_out, pre_pool

def parse_option():
    parser = argparse.ArgumentParser('Evaluation script')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset to evaluate on')

    # dataset
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'ResNet50', 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'resnet18'],  # Add resnet18 here
                        help='model architecture')

    parser.add_argument('--model_path', type=str, default=r"C:\Users\ashaz\OneDrive\Desktop\supervised-contrastive-kd-main\save\models\cifar10\resnet18_cifar10\resnet18_best.pth",
                        help='Path to the model checkpoint')

    opt = parser.parse_args()
    return opt

def load_checkpoint(model, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Try common keys for state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # If model is wrapped in DataParallel, keys should start with 'module.'
    # Add prefix if missing
    if list(state_dict.keys())[0].startswith('module.'):
        model.load_state_dict(state_dict)
    else:
        # Add 'module.' prefix for DataParallel model
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = 'module.' + k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)

def main():
    opt = parse_option()

    # Load dataset
    train_loader, val_loader, n_cls = choose_dataset(dataset=opt.dataset,
                                                     batch_size=opt.batch_size,
                                                     num_workers=opt.num_workers)

    # Initialize model and wrap in DataParallel
    if opt.model == 'resnet18':
        model = ResNet18(num_classes=n_cls)  # Load ResNet18 model
    else:
        model = model_dict[opt.model](num_classes=n_cls)
    
    model = torch.nn.DataParallel(model)

    # Load checkpoint weights into model
    load_checkpoint(model, opt.model_path)

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    model.eval()

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # Evaluate model
    accuracy = validate(val_loader, model, criterion, print_freq=100)
    print(f"Evaluation result - Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
