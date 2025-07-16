import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import ResNet
class SimCLR(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = ResNet(
            x_channels=config["model"]["encoder_params"]["input_channels"],
            layers=config["model"]["encoder_params"]["layer_channels"]
        )


        self.projection_head = nn.Sequential(
            nn.Linear(config["model"]["projection_head"]["input_dim"],
              config["model"]["projection_head"]["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(config["model"]["projection_head"]["hidden_dim"],
                    config["model"]["projection_head"]["output_dim"])
        )
    
    def forward(self, x: torch.Tensor):
        return self.projection_head(self.encoder(x))
        

if __name__ == '__main__':
    import yaml
    x = torch.rand(2, 3, 256, 256)

    with open ('config/config.yml', 'r') as file:
        config = yaml.safe_load(file)

    model = SimCLR(config=config)
    out = model(x)

    num_params = sum(p.numel() for p in model.parameters())

    print(out.shape)
