from gelgenie.segmentation.networks.DAUNet.models_DAUNet import DAUNet

class daunet(DAUNet):
    def __init__(self, in_channels=1, classes=2, **kwargs):
        super().__init__(in_channels=in_channels, num_classes=classes)
        self.n_channels = in_channels
        self.n_classes = classes
