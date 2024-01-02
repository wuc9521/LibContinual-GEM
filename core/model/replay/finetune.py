import torch
from torch import nn

class Finetune(nn.Module):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):
        super().__init__()
        self.backbone = backbone # torchvision model
        self.feat_dim = feat_dim # feature dimension
        self.num_class = num_class # number of classes
        self.classifier = nn.Linear(feat_dim, num_class) # linear classifier
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean') # loss function
        self.device = kwargs['device'] # device
    
    def observe(self, data):
        x, y = data['image'], data['label'] # get data
        x = x.to(self.device)
        y = y.to(self.device)
        logit = self.classifier(self.backbone(x)['features'])    
        loss = self.loss_fn(logit, y)

        pred = torch.argmax(logit, dim=1)

        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0), loss

    def inference(self, data):
        x, y = data['image'], data['label']
        x = x.to(self.device)
        y = y.to(self.device)

        logit = self.classifier(self.backbone(x)['features'])  
        pred = torch.argmax(logit, dim=1)
        acc = torch.sum(pred == y).item()
        return pred, acc / x.size(0)

    def forward(self, x):
        return self.classifier(self.backbone(x)['features'])  
    
    def before_task(self, task_idx, buffer, train_loader, test_loaders): pass

    def after_task(self, task_idx, buffer, train_loader, test_loaders): pass