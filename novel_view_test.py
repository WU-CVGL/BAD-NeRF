import nerf
import torch.nn


class Model(nerf.Model):
    def __init__(self, se3_start, graph):
        super().__init__()
        self.start = se3_start
        self.graph_fixed = graph

    def build_network(self, args):
        self.graph_fixed.se3_sharp = torch.nn.Embedding(self.start.shape[0], 6)  # 22和25
        self.graph_fixed.se3_sharp.weight.data = torch.nn.Parameter(self.start)

        return self.graph_fixed

    def setup_optimizer(self, args):
        grad_vars_se3 = list(self.graph_fixed.se3_sharp.parameters())
        self.optim_se3_sharp = torch.optim.Adam(params=grad_vars_se3, lr=args.lrate)

        return self.optim_se3_sharp