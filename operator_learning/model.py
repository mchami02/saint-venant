from neuralop.models import FNO
from deepxde.nn.pytorch import DeepONetCartesianProd
from models.fno_wrapper import fno_custom_freqs
from models.wno import WNO2d
from models.lno import LNOWrapper
from models.deeponet import DeepONetWrapper
# DeepXDE sets torch default device to cuda, but this breaks DataLoader with num_workers > 0
# Reset it to None/cpu to avoid generator device mismatch
import torch
torch.set_default_device(None)


def create_model(args):
    if args.model == "FNO":
        model = FNO(
        n_modes=(16, 4),        # modes in (time, space) dimensions
        hidden_channels=16,       # network width
        in_channels=args.in_channels - 2,           # density + time + space
        out_channels=args.out_channels,          # predicted density
        n_layers=4               # number of FNO layers
        )
    elif args.model == "FNO1d":
        model = FNO(
        n_modes=(16,),        # modes in (time) dimension
        hidden_channels=16,       # network width
        in_channels=1,           # density + time
        out_channels=1,          # predicted density
        n_layers=4               # number of FNO layers
        )
    elif args.model == "FNOPersonalized":
        model = fno_custom_freqs(
            n_modes=(8, 16),
            hidden_channels=16,
            in_channels=args.in_channels - 2,
            out_channels=args.out_channels,
            n_layers=4
        )
    elif args.model == "DeepONet":
        # Calculate proper branch network input size
        # Branch receives: initial condition (n_features*nx) + boundaries (2*n_features*nt)
        n_features = args.in_channels - 2  # Exclude coordinate channels
        branch_input_size = n_features * (args.nx + 2 * args.nt)
        
        model = DeepONetWrapper(
        nt=args.nt,
        nx=args.nx,
        dt=args.dt,
        dx=args.dx,
        n_features=n_features,
        layer_sizes_branch = [branch_input_size, 256, 512, 1024, 512, 256, 128],  # Deeper network for better learning
        layer_sizes_trunk = [2, 256, 256, 256, 128],  # 2 for (t, x) coordinates
        activation = "relu",
        kernel_initializer = "Glorot normal",
        num_outputs = args.out_channels
        )
    elif args.model == "WNO":
        model = WNO2d(
            width=8,
            level=3,
            layers=2,
            size=[args.nt, args.nx],
            wavelet='db4',
            in_channel=3,
            grid_range=[0.0, 1.0],
        )
    elif args.model == "LNO":
        n_features = args.in_channels - 2  # Exclude coordinate channels
        model = LNOWrapper(
            nt=args.nt,
            nx=args.nx,
            dt=args.dt,
            dx=args.dx,
            in_channels=n_features,
            out_channels=args.out_channels,
            n_block=8,          # Number of attention blocks
            n_mode=256,         # Number of latent modes  
            n_dim=256,          # Hidden dimension
            n_head=8,           # Number of attention heads
            n_layer=2,          # Number of MLP layers
            attn="Attention_Vanilla",  # Attention type
            act="GELU"          # Activation function
        )
    else:
        raise ValueError(f"Model {args.model} not supported")
    return model
