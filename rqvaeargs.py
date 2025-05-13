from quantize import QuantizeForwardMode

class RqvaeArgs:
    def __init__(self):
        self.input_dim = 1024
        self.embed_dim = 1024
        self.hidden_dims = [2048, 1024]
        self.codebook_size = 64
        self.codebook_sim_vq = True
        self.codebook_normalization = True
        self.do_kmeans_init = False
        self.num_layers = 3
        self.commitment_weight = 0.25
        self.quantize_mode = QuantizeForwardMode.GUMBEL_SOFTMAX
        self.epochs = 10,
        self.batch_size = 16
        self.lr = 1e-4
        self.eps = 1e-5
        self.data_path = ""
        self.use_ratio = 1.0
        self.train_split = 0.9
        self.val_split = 0.05
        self.start_step = 0
        self.save_steps_interval = 20000
        self.checkpoint_path = ""
        self.valid_interval = 2
        self.device = "gpu"
        self.mode = "train"
        self.plot_train_curve = True
        self.model_on_path = ""