from quantize import QuantizeForwardMode

class RqvaeArgs:
    def __init__(self):
        self.input_dim = 256,
        self.embed_dim = 256,
        self.hidden_dims = [512, 256],
        self.codebook_size = 64,
        self.codebook_sim_vq = True,
        self.codebook_normalization = True,
        self.do_kmeans_init = False,
        self.num_layers = 3,
        self.commitment_weight = 0.25,
        self.quantize_mode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        self.epochs = 10,
        self.batch_size = 16
        self.lr = 1e-4
        self.eps = 1e-5