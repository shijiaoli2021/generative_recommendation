import paddle.optimizer

import rqvae
from rqvae_dataset import ItemDataset
from rqvae import *
from quantize import QuantizeForwardMode
from rqvaeargs import RqvaeArgs
from util import *
from paddle.io import DataLoader
from a_trainer import AbstractTrainer
import matplotlib.pyplot as plt


def plot_curve(y_list:list, label_list:list, step_interval, save_path:str):

    for i in range(len(y_list)):
        plt.plot( y_list[i][0:len(y_list):step_interval], label=label_list[i], linewidth=2)
    plt.legend()
    plt.savefig(save_path)

def buildDataLoader(args:RqvaeArgs):

    item_data_dict = read_ad_data(args.data_path, args.use_ratio)
    item_values = list(item_data_dict.values())
    total = len(item_data_dict)
    train_size = int(total * args.train_split)
    val_size = int(total * args.val_split)

    train_dataset = ItemDataset(data=item_values[:train_size])
    val_dataset = ItemDataset(data=item_values[train_size:train_size+val_size])
    test_dataset = ItemDataset(data=item_values[train_size+val_size:])

    return (DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size),
            DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size),
            DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size))

class Trainer:
    def __init__(
        self,
        model: RqVae,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: paddle.optimizer.Optimizer,
        args:RqvaeArgs
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.args = args
        self.device = args.device
        self.steps = args.start_step
        self.save_steps_interval = args.save_steps_interval
        self.train_loss = []
        self.rqvae_loss = []
        self.reconstruct_loss = []

    def _train_one_epoch(self, epoch):
        train_loss = []
        rqvae_loss = []
        reconstruct_loss = []
        with tqdm(self.train_loader, desc=f"epoch:{epoch}") as tq:
            for batch in tq:

                batch = batch.to(self.device)

                rqvae_out = self.model.forward(batch)

                rqvae_out.loss.backward()

                self.optimizer.step()

                self.steps += 1

                # update loss on tq
                if self.steps % 2 == 0:
                    tq.set_postfix_str(f"rq_loss:{rqvae_out.rqvae_loss:3f}, recon_loss:{rqvae_out.reconstruction_loss:3f}")

                train_loss.append(rqvae_out.loss.detach().item())
                rqvae_loss.append(rqvae_out.rqvae_loss.detach().item())
                reconstruct_loss.append(rqvae_out.reconstruction_loss.detach().item())

                # save model
                self._save_in_train()

        self.train_loss += train_loss
        self.rqvae_loss += rqvae_loss
        self.reconstruct_loss += reconstruct_loss
        print(f"train {epoch} epoch over, mean loss:{sum(train_loss) / len(train_loss)},"
              f" rqvae mean loss:{sum(rqvae_loss) / len(rqvae_loss)},"
              f" reconstruct_loss:{sum(reconstruct_loss) / len(reconstruct_loss)}")



    def _save_in_train(self):
        if self.steps % self.save_steps_interval == 0:
            print(f"have trained for {self.steps} steps, save now...")
            self._save_model()
            print("save successfully...")

    def _save_model(self):
        save_name = f"RqVae_{self.steps}.pth"
        paddle.save({"model_parm": self.args, "state_dict": self.model.state_dict()},
                    self.args.checkpoint_path + save_name)

    def train(self):

        # model train mode
        self.model.train()
        print("train begin...")
        for epoch in range(self.args.epochs):
            self._train_one_epoch(epoch)
            self.valid_stage(epoch)
        print("train over, test begins...")
        self.test()
        self._save_model()

    def valid_stage(self, epoch):
        if epoch % self.args.valid_interval == 0:
            self.valid()

    def valid(self):
        self._eval_preprocess(mode='val')

    def test(self):
        self._eval_preprocess(mode='test')

    def _eval_preprocess(self, mode:str):
        if mode == 'val':
            dataloader = self.val_loader
        elif mode == 'test':
            dataloader = self.test_loader
        else:
            raise Exception(f"unsupported mode for {mode}...")
        print(f"{mode} begin...")
        train_loss = []
        rqvae_loss = []
        reconstruct_loss = []
        with tqdm(dataloader) as tq:
            for batch in self.val_loader:
                rqvae_out = self.model(batch)

                # update loss on tq
                if self.steps % 2 == 0:
                    tq.set_postfix_str(
                        f"rq_loss:{rqvae_out.rqvae_loss:3f}, recon_loss:{rqvae_out.reconstruction_loss:3f}")

                train_loss.append(rqvae_out.loss.detach().item())
                rqvae_loss.append(rqvae_out.rqvae_loss.detach().item())
                reconstruct_loss.append(rqvae_out.reconstruction_loss.detach().item())
        print(f"{mode} over, mean loss:{sum(train_loss) / len(train_loss)},"
              f" rqvae mean loss:{sum(rqvae_loss) / len(rqvae_loss)},"
              f" reconstruct_loss:{sum(reconstruct_loss) / len(reconstruct_loss)}")


if __name__ == '__main__':

    args = RqvaeArgs()

    if args.model_on_path is not None and args.model_on_path != "":
        checkpoint = paddle.load(args.model_on_path)
        model = rqvae.RqVae(checkpoint["model_parm"])
        model.load_dict(checkpoint["state_dict"])
        model = model.to(args.device)
    else:
        # model
        model = rqvae.RqVae(args).to(args.device)

    # optimizer
    optimizer = paddle.optimizer.Adagrad(parameters=model.parameters(), learning_rate=args.lr, epsilon=args.eps)

    # data
    train_loader, val_loader, test_loader = buildDataLoader(args)

    # trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, optimizer, args)

    if args.mode == 'train':
        trainer.train()
        if args.plot_train_curve:
            plot_curve([trainer.train_loss, trainer.rqvae_loss, trainer.reconstruct_loss],
                       ["train_loss", "rqvae_loss", "reconstruct_loss"],
                       step_interval = 3,
                       save_path=f"./figures/rqvae_cs{args.codebook_size}_cn{args.num_layers}_{args.epochs}.png")

    elif args.mode == 'test':
        trainer.test()
    else:
        raise Exception(f"unsupported mode for {args.mode}")