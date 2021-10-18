import os
import torch
import torchvision

# Bilinear Pooling operats between two different groups
class Inter_GBP(torch.nn.Module):
    def __init__(self, backbone='resnet50', channel_num=2048, sample_group=1024, classes=200, freeze=True):
        torch.nn.Module.__init__(self)
        self.sample_group = sample_group                        # the total number of groups
        self.classes = classes                                  # the classes of dataset
        self.channel_num = channel_num                          # the number of channels
        self.size = int(self.channel_num/self.sample_group)   

        if backbone=='resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=True)
        elif backbone=='resnet101':
            self.backbone = torchvision.models.resnet101(pretrained=True)
        elif backbone=='resnet152':
            self.backbone = torchvision.models.resnet152(pretrained=True)
        elif backbone=='vgg16':
            self.backbone = torchvision.models.vgg16(pretrained=True).features 
            self.features_vgg = torch.nn.Sequential(*list(self.backbone.children())[:-1])     
            self.fc = torch.nn.Linear(size*size*self.sample_group, self.classes)

        if backbone in ['resnet50','resnet101','resnet152']：
            self.features = torch.nn.Sequential(*list(self.backbone.children())[:-2])     # [2048,14,14]
        elif backbone=='vgg16':
            self.features = torch.nn.Sequential(*list(self.backbone.children())[:-1])     # [512,24,24]

        self.fc = torch.nn.Linear(int(self.size*self.size*self.sample_group/2), self.classes)

        torch.nn.init.xavier_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)

        if freeze:
            for param in self.features.parameters():   
                param.requires_grad = False

    # Bilinear pooling between two different groups
    def FBP(self, conv1, conv2):
        N,C,H,W = conv1.size()
        conv1 = torch.reshape(conv1, (N, self.size, H * W))
        conv2 = torch.reshape(conv2, (N, self.size, H * W))
        conv1 = torch.bmm(conv1, torch.transpose(conv2, 1, 2)) / (H * W)
        conv1 = torch.reshape(conv1, (N, self.size * self.size))
        conv1 = torch.sqrt(conv1 + 1e-5)
        conv1 = torch.nn.functional.normalize(conv1)
        return conv1

    def forward(self, X):
        N,C,H,W = X.size()
        X_conv = self.features(X)                      

        for i in range(0, self.channel_num, self.size*2):
            X_branch_temp1 = X_conv[:, i:i+self.size, :, :]
            X_branch_temp2 = X_conv[:, i+self.size:i+int(self.size*2), :, :]
            X_branch_temp = self.FBP(X_branch_temp1, X_branch_temp2)
            if(i==0):
                X_branch = X_branch_temp
            else:
                X_branch = torch.cat([X_branch, X_branch_temp],dim = 1)

        assert X_branch.size() == (N, int(self.size*self.size*self.sample_group/2))
        X = self.fc(X_branch)
        assert X.size() == (N, self.classes)
        return X


# Bilinear Pooling operats between two same groups 
class Intra_GBP(torch.nn.Module):
    def __init__(self, backbone='resnet50', channel_num=2048, sample_group=1024, classes=200, freeze=True):
        torch.nn.Module.__init__(self)
        self.sample_group = sample_group                    # the total number of groups
        self.classes = classes                              # the classes of dataset
        self.channel_num = channel_num                          # the number of channels
        self.size = int(self.channel_num/self.sample_group)   

        if backbone=='resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=True)
        elif backbone=='resnet101':
            self.backbone = torchvision.models.resnet101(pretrained=True)
        elif backbone=='resnet152':
            self.backbone = torchvision.models.resnet152(pretrained=True)
        elif backbone=='vgg16':
            self.backbone = torchvision.models.vgg16(pretrained=True).features 
            self.features_vgg = torch.nn.Sequential(*list(self.backbone.children())[:-1])     
            self.fc = torch.nn.Linear(size*size*self.sample_group, self.classes)

        if backbone in ['resnet50','resnet101','resnet152']：
            self.features = torch.nn.Sequential(*list(self.backbone.children())[:-2])     # [2048,14,14]
        elif backbone=='vgg16':
            self.features = torch.nn.Sequential(*list(self.backbone.children())[:-1])     # [512,24,24]

        self.fc = torch.nn.Linear(int(self.size*self.size*self.sample_group), self.classes)

        torch.nn.init.xavier_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)

        if freeze:
            for param in self.features.parameters():   
                param.requires_grad = False

    # Bilinear pooling
    def FBP(self, conv1, conv2):
        N,C,H,W = conv1.size()
        conv1 = torch.reshape(conv1, (N, self.size, H * W))
        conv2 = torch.reshape(conv2, (N, self.size, H * W))
        conv1 = torch.bmm(conv1, torch.transpose(conv2, 1, 2)) / (H * W)
        conv1 = torch.reshape(conv1, (N, self.size * self.size))
        conv1 = torch.sqrt(conv1 + 1e-5)
        conv1 = torch.nn.functional.normalize(conv1)
        return conv1

    def forward(self, X):
        N,C,H,W = X.size()
        X_conv = self.features(X)     

        for i in range(0, self.channel_num, self.size):
            X_branch_temp1 = X_conv[:, i:i+self.size, :, :]
            X_branch_temp = self.FBP(X_branch_temp1, X_branch_temp1)
            if(i==0):
                X_branch = X_branch_temp
            else:
                X_branch = torch.cat([X_branch, X_branch_temp],dim = 1)
        assert X_branch.size() == (N, int(self.size*self.size*self.sample_group))
        X = self.fc(X_branch)
        assert X.size() == (N, self.classes)
        return X
