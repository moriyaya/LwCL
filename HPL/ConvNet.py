
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def conv3x3(in_channels, out_channels,activation=nn.ReLU(inplace=True)):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.MaxPool2d(2),
            nn.BatchNorm2d(out_channels, momentum=1., affine=True,
                           track_running_stats=False # When this is true is called the "transductive setting"
                            ), activation
                    )

class FullyConnectedLayer(nn.Module):
    def __init__(self, num_layer=2):
        super(FullyConnectedLayer, self).__init__()
        '''
        self.classifier = nn.Sequential(
                                       nn.Flatten(),
                                       nn.Linear(in_shape, out_features))
        self.hidden_size = self.hidden_size
        '''
        self.fc_net = nn.Sequential(nn.Linear(1,64),nn.ReLU(inplace=True),nn.LayerNorm(normalized_shape=64))
        for j in range(num_layer-1):
            self.fc_net = nn.Sequential(self.fc_net, nn.Linear(64,64),nn.ReLU(inplace=True),nn.LayerNorm(normalized_shape=64)
                                                                                                           )
    def forward(self, inputs, params=None):
        #features = inputs.view((inputs.size(0), -1))
        logits = self.fc_net(inputs)
        return logits

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

class TaskFullyConnectedLayer(nn.Module):
    def __init__(self,num_layer=1, task_conv=0):
        super(TaskFullyConnectedLayer, self).__init__()
        '''
        self.classifier = nn.Sequential(
                                       nn.Flatten(),
                                       nn.Linear(in_shape, out_features))
        self.hidden_size = self.hidden_size
        '''
        if num_layer>1:
            self.classifier = nn.Linear(64,1)

        self.classifier = nn.Sequential(nn.Linear(64,1))

    def forward(self, inputs, params=None):
        #features = inputs.view((inputs.size(0), -1))
        logits = self.classifier(inputs)
        return logits

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param



class TaskLinearLayer(nn.Module):
    def __init__(self, in_shape, out_features,hidden_size=32,task_conv=0,dfc=True):
        super(TaskLinearLayer, self).__init__()
        self.in_shape = in_shape
        self.out_features = out_features

        if task_conv ==0 and not dfc:
            self.classifier  = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_shape, out_features))
        elif dfc:
            self.classifier  = nn.Sequential(conv3x3(hidden_size, hidden_size,activation=nn.Softplus()), nn.Flatten(), nn.Linear(128, out_features))
        else:
            self.classifier = conv3x3(hidden_size, hidden_size)
            for j in range(task_conv-1):
                self.classifier = nn.Sequential(self.classifier, conv3x3(hidden_size, hidden_size,activation=nn.Softplus()))
            self.classifier = nn.Sequential(self.classifier,
                                            nn.Flatten(), nn.Linear(in_shape, out_features))

    def forward(self, inputs, params=None):
        #features = inputs.view((inputs.size(0), -1))
        logits = self.classifier(inputs)
        return logits

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

class  ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, in_channels, out_features, hidden_size=32,device=None,task_conv=0):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        assert task_conv >= 0, "Wrong call for task nets!"
        self.features = conv3x3(in_channels, hidden_size)
        for i in range(3-task_conv):
            self.features = nn.Sequential(self.features, conv3x3(hidden_size, hidden_size))

    def forward(self, inputs, params=None):
        features = self.features(inputs)
        return features

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param


