from torch.nn import (
    Module,
    SyncBatchNorm,
    Sequential,
    Linear,
    BatchNorm1d,
    LeakyReLU,
    Tanh,
)
from torch import split, cat, load, save
from .featureExtractor import Encoder, Decoder
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
from .fitnessNet import (
    BasicFitnessNet,
    ResFitnessNet,
    SpatialConcatResFitnessNet,
    SpatialConcatResFitnessNetV2
)
from .generatorNet import (
    BaseGenerator,
    DeepThinGenerator,
    DeeperNoBatchnormGenerator,
    DeeperGenerator
)


class GripperDesigner(Module):
    def __init__(self, embedding_dim,
                 designer_weight_decay, designer_lr,
                 vae_weight_decay, vae_lr, device,
                 fitness_class_str=None,
                 generator_class_str=None,
                 checkpoint_dict=dict(),
                 load_fitness_net=True,
                 optimize_objectives=['success', 'stability', 'robustness'],
                 use_1_robustness=False):
        super().__init__()
        self.designer_lr = designer_lr
        self.designer_weight_decay = designer_weight_decay
        self.distributed = False

        self.vae_weight_decay = vae_weight_decay
        self.vae_lr = vae_lr
        self.embedding_dim = embedding_dim
        self.device = device
        self.gn_encoder = Encoder(self.embedding_dim).to(self.device)
        self.decoder = Decoder(self.embedding_dim).to(self.device)

        if fitness_class_str is None or fitness_class_str == "ResFitnessNet":
            fitness_class = ResFitnessNet
        elif fitness_class_str == "SpatialConcatResFitnessNet":
            fitness_class = SpatialConcatResFitnessNet
        elif fitness_class_str == "SpatialConcatResFitnessNetV2":
            fitness_class = SpatialConcatResFitnessNetV2
        elif fitness_class_str == "BasicFitnessNet":
            fitness_class = BasicFitnessNet
        else:
            raise NotImplementedError

        if generator_class_str is None \
                or generator_class_str == "BaseGenerator":
            generator_class = BaseGenerator
        elif generator_class_str == "DeepThinGenerator":
            generator_class = DeepThinGenerator
        elif generator_class_str == "DeeperNoBatchnormGenerator":
            generator_class = DeeperNoBatchnormGenerator
        elif generator_class_str == "DeeperGenerator":
            generator_class = DeeperGenerator
        else:
            raise NotImplementedError

        self.fitness_net = None
        self.fn_optimizer = None
        self.optimize_objectives = optimize_objectives
        self.use_1_robustness = use_1_robustness
        if load_fitness_net:
            self.fitness_net = fitness_class(
                optimize_objectives, use_1_robustness).to(self.device)
            self.fn_optimizer = Adam(
                self.fitness_net.parameters(),
                lr=self.designer_lr,
                weight_decay=self.designer_weight_decay)

        self.generator_net = Sequential(
            Linear(self.embedding_dim, 4096),
            BatchNorm1d(4096),
            LeakyReLU(),
            #
            Linear(4096, 4096),
            BatchNorm1d(4096),
            LeakyReLU(),
            #
            Linear(4096, self.embedding_dim * 2),
            BatchNorm1d(self.embedding_dim * 2),
            Tanh()
        ).to(self.device)

        self.vae_optimizer = Adam(
            list(self.decoder.parameters()) +
            list(self.gn_encoder.parameters()),
            lr=self.vae_lr,
            weight_decay=self.vae_weight_decay)
        self.gn_optimizer = Adam(
            list(self.generator_net.parameters()) +
            list(self.gn_encoder.parameters()),
            lr=self.designer_lr,
            weight_decay=self.designer_weight_decay)
        self.stats = {
            'epochs': 0,
            'update_steps': 0,
            'cycles': 0,
            'fn_update_steps': 0,
            'gn_update_steps': 0,
        }

        if load_fitness_net and checkpoint_dict["fn_checkpoint_path"] is not None:
            self.load_fn_checkpoint(checkpoint_dict["fn_checkpoint_path"])
        if checkpoint_dict["gn_checkpoint_path"] is not None:
            self.load_gn_checkpoint(checkpoint_dict["gn_checkpoint_path"])
        elif checkpoint_dict["ae_checkpoint_path"] is not None:
            self.load_ae_checkpoint(checkpoint_dict["ae_checkpoint_path"])

    def distribute_net(self, net, device):
        net = net.to(self.device)
        return DDP(
            SyncBatchNorm.convert_sync_batchnorm(net).to(self.device),
            device_ids=[device])

    def distribute(self, device):
        self.gn_encoder = self.distribute_net(self.gn_encoder, device)
        self.decoder = self.distribute_net(self.decoder, device)
        self.fitness_net = self.distribute_net(self.fitness_net, device)
        self.generator_net = self.distribute_net(self.generator_net, device)
        self.distributed = True

    def create_fingers(self, grasp_object):
        grasp_object = grasp_object.to(self.device)
        grasp_object = self.gn_encoder(grasp_object)[0]
        fingers_emb = self.generator_net(grasp_object)
        left_finger_emb, right_finger_emb = split(
            fingers_emb, self.embedding_dim, dim=1)
        return self.decoder(left_finger_emb),\
            self.decoder(right_finger_emb)

    def evaluate_fingers(self, grasp_object, left_finger, right_finger):
        return self.fitness_net(
            cat((
                grasp_object.to(self.device),
                left_finger.to(self.device),
                right_finger.to(self.device)
            ), dim=1))

    def encode_decode(self, input):
        z = self.gn_encoder.reparameterize(input.to(self.device))
        return self.decoder(z)

    def __load_ae_checkpoint(self, nets):
        self.gn_encoder.load_state_dict(nets['gn_encoder'])
        self.decoder.load_state_dict(nets['decoder'])

    def load_ae_checkpoint(self, path):
        print(f'[GripperDesigner] loading autoencoder network from {path}')
        checkpoint = load(path, map_location=self.device)
        nets = checkpoint['networks']
        self.__load_ae_checkpoint(nets)
        optimizers = checkpoint['optimizers']
        self.gn_optimizer.load_state_dict(optimizers['gn_optimizer'])
        self.stats = checkpoint['stats']
        print("Loaded generator autoencoder and optimizer from path: ", path)

    def load_gn_checkpoint(self, path):
        print(f'[GripperDesigner] loading generator network from {path}')
        checkpoint = load(path, map_location=self.device)
        nets = checkpoint['networks']
        self.__load_ae_checkpoint(nets)
        self.generator_net.load_state_dict(nets['generator'])
        optimizers = checkpoint['optimizers']
        self.gn_optimizer.load_state_dict(optimizers['gn_optimizer'])
        self.stats = checkpoint['stats']
        print("Loaded generator network and optimizer from path: ", path)

    def load_fn_checkpoint(self, path):
        print(f'[GripperDesigner] loading fitness network from {path}')
        checkpoint = load(path, map_location=self.device)
        self.fitness_net.load_state_dict(checkpoint['networks']['fitness'])
        self.fn_optimizer.load_state_dict(checkpoint['optimizers']['fn_optimizer'])
        self.stats = checkpoint['stats']
        print("Loaded fitness network and optimizer from path: ", path)

    def get_optimizer_state_dicts(self):
        return {
            'fn_optimizer': self.fn_optimizer.state_dict(),
            'gn_optimizer': self.gn_optimizer.state_dict()
        }

    def get_network_state_dicts(self):
        if self.distributed:
            return {
                'gn_encoder': self.gn_encoder.module.state_dict(),
                'decoder': self.decoder.module.state_dict(),
                'fitness': self.fitness_net.module.state_dict(),
                'generator': self.generator_net.module.state_dict()
            }
        else:
            return {
                'gn_encoder': self.gn_encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
                'fitness': self.fitness_net.state_dict(),
                'generator': self.generator_net.state_dict()
            }

    def save(self, epoch, checkpoint_path):
        self.stats['epochs'] = epoch
        save({
            'stats': self.stats,
            'networks': self.get_network_state_dicts(),
            'optimizers': self.get_optimizer_state_dicts()
        }, checkpoint_path)
        print(f'[GripperDesigner] Saved checkpoint to {checkpoint_path}')
