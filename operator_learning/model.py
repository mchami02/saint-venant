import torch
from models.deeponet import DeepONetWrapper
from models.encoder_decoder import EncoderDecoder
from models.fno_cnn import FNOCNNWrapper
from models.fno_denoiser import FNODenoiserWrapper
from models.fno_wrapper import fno_custom_freqs
from models.lno import LNOWrapper
from models.moe_fno import MoEFNO
from models.wave_front_pred import WaveFront
from models.wave_front_router import WaveFrontFNO
from models.wno import WNO2d
from neuralop.models import FNO

torch.set_default_device(None)

class OperatorModel(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.metadata = kwargs
        self.model = model(**kwargs)

    def forward(self, x):
        return self.model(x)


def create_model(args, device):
    if args.model == "FNO":
        model = OperatorModel(FNO,
        n_modes=(64, 16),        # modes in (time, space) dimensions
        hidden_channels=64,       # network width
        in_channels=args.in_channels,           # density + time + space
        out_channels=args.out_channels,          # predicted density
        n_layers=4               # number of FNO layers
        )
    elif args.model == "FNO1d":
        model = OperatorModel(FNO,
        n_modes=(16,),        # modes in (time) dimension
        hidden_channels=16,       # network width
        in_channels=1,           # density + time
        out_channels=1,          # predicted density
        n_layers=4               # number of FNO layers
        )
    elif args.model == "FNOPersonalized":
        model = OperatorModel(fno_custom_freqs,
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
        
        model = OperatorModel(DeepONetWrapper,
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
        model = OperatorModel(WNO2d,
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
        model = OperatorModel(LNOWrapper,
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
    # elif OperatorModel(args.model == "GNN"):
    #     n_features = args.in_channels - 2  # Exclude coordinate channels
    #     model = MaskedGridPredictor(
    #         nt=args.nt,
    #         nx=args.nx,
    #         dt=args.dt,
    #         dx=args.dx,
    #         in_channels=n_features,
    #         out_channels=args.out_channels,
    #         hidden_dim=args.hidden_channels,
    #         num_heads=4,
    #         num_encoder_layers=args.n_layers,
    #         num_decoder_layers=args.n_layers,
    #         dropout=0.0,
    #         relative_emb=True,
    #         mask_value=-1.0
    #     )
    elif args.model == "FNOCNN":
        model = OperatorModel(FNOCNNWrapper,
            in_channels=args.in_channels - 2,  # Exclude coordinate channels, like FNO
            out_channels=args.out_channels,
            fno_modes=(64, 16),
            fno_hidden_channels=64,
            fno_layers=4,
            cnn_hidden_channels=64,
            cnn_layers=6,
            cnn_kernel_size=3,
            skip_connection=True,
            strip_coords=False,  # Coords already excluded above
        )
    elif args.model == "FNODenoiser":
        model = OperatorModel(FNODenoiserWrapper,
            in_channels=args.in_channels - 2,  # Exclude coordinate channels, like FNO
            out_channels=args.out_channels,
            fno_modes=(64, 16),
            fno_hidden_channels=64,
            fno_layers=4,
            cnn_hidden_channels=64,
            cnn_layers=6,
            cnn_kernel_size=3,
            skip_connection=True,
            strip_coords=False,  # Coords already excluded above
        )
    elif args.model == "EncoderDecoder":
        model = OperatorModel(EncoderDecoder,
            hidden_dim=64,
            layers_encoder=2,
            decoder_type="axial",
            layers_decoder=2,
            layers_gnn=0,
        )
    elif args.model == "EncoderDecoderCross":
        model = OperatorModel(EncoderDecoder,
            hidden_dim=64,
            layers_encoder=2,
            decoder_type="cross",
            layers_decoder=2,
            layers_gnn=0,
        )
    elif args.model == "MOEFNO":
        model = OperatorModel(MoEFNO,
            n_modes=(16, 8),
            hidden_channels=32,
            n_experts=8,
            in_channels=3,
            out_channels=1,
            n_layers=4,
            router_hidden_dim=32,
            router_num_layers=2,
            router_num_heads=4,
            top_k=2,
        )
    elif args.model == "WaveFrontFNO":
        model = OperatorModel(WaveFrontFNO,
            n_experts=5,
            n_modes=(16, 8),
            n_layers=4,
            hidden_dim=32,
            in_channels=3,
            num_encoder_layers=2,
            num_heads=8,
            max_fronts=5,
            boundary_sharpness=10.0,
        )
    elif args.model == "WaveFront":
        model = OperatorModel(WaveFront,
            n_modes=(16, 8),
            hidden_dim=32,
            in_channels=3,
            out_channels=1,
            n_layers=4,
            front_threshold=0.5,
        )
    else:
        raise ValueError(f"Model {args.model} not supported")
    return model.to(device)
