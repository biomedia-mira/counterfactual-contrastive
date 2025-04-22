import copy

HPARAMS_REGISTRY = {}


class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)


embed = Hparams()
embed.input_channels = 1
embed.lr = 1e-3
embed.wd = 1e-3
embed.lr_warmup_steps = 100
embed.bottleneck = 4
embed.cond_prior = True
embed.z_max_res = 128
embed.z_dim = [48, 30, 24, 18, 12, 6, 1]
embed.input_res = 224
embed.enc_arch = "224b2d2,112b4d2,56b9d2,28b14d2,14b14d2,7b9d7,1b5"
embed.dec_arch = "1b5,7b10,14b15,28b15,56b10,112b5,224b3"
embed.widths = [32, 64, 96, 128, 160, 192, 512]
embed.parents_x = ["scanner"]
embed.context_dim = 5
embed.z_max_res = 96
embed.eval_freq = 1
embed.grad_clip = 350
embed.grad_skip = 25000
embed.beta = 1.0
embed.accu_steps = 2
embed.bias_max_res = 64
embed.x_like = "fixed_dgauss"
embed.std_init = 1e-2
embed.epochs = 30
embed.bs = 16
HPARAMS_REGISTRY["embed"] = embed

embedbad = copy.deepcopy(embed)
embedbad.epochs = 1
embedbad.beta = 2.0
HPARAMS_REGISTRY["embedbad"] = embedbad

padchest = copy.deepcopy(embed)
padchest.parents_x = ["scanner", "sex"]
padchest.context_dim = 2
padchest.beta = 3.0
padchest.epochs = 130
HPARAMS_REGISTRY["padchest"] = padchest


def setup_hparams(parser):
    hparams = Hparams()
    args = parser.parse_known_args()[0]
    valid_args = set(args.__dict__.keys())
    hparams_dict = HPARAMS_REGISTRY[args.hps].__dict__
    for k in hparams_dict.keys():
        if k not in valid_args:
            raise ValueError(f"{k} not in default args")
    parser.set_defaults(**hparams_dict)
    hparams.update(parser.parse_known_args()[0].__dict__)
    return hparams


def add_arguments(parser):
    parser.add_argument("--bs", help="batch size", type=str, default=16)
    parser.add_argument("--exp_name", help="Experiment name.", type=str, default="")
    parser.add_argument(
        "--data_dir", help="Data directory to load form.", type=str, default=""
    )
    parser.add_argument("--hps", help="hyperparam set.", type=str, default="ukbb64")
    parser.add_argument(
        "--resume", help="Path to load checkpoint.", type=str, default=""
    )
    parser.add_argument("--seed", help="Set random seed.", type=int, default=7)
    parser.add_argument(
        "--deterministic",
        help="Toggle cudNN determinism.",
        action="store_true",
        default=False,
    )
    # training
    parser.add_argument("--epochs", help="Training epochs.", type=int, default=5000)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=1e-3)
    parser.add_argument(
        "--lr_warmup_steps", help="lr warmup steps.", type=int, default=100
    )
    parser.add_argument("--wd", help="Weight decay penalty.", type=float, default=0.01)
    parser.add_argument(
        "--betas",
        help="Adam beta parameters.",
        nargs="+",
        type=float,
        default=[0.9, 0.9],
    )
    parser.add_argument(
        "--ema_rate", help="Exp. moving avg. model rate.", type=float, default=0.999
    )
    parser.add_argument(
        "--input_res", help="Input image crop resolution.", type=int, default=64
    )
    parser.add_argument(
        "--input_channels", help="Input image num channels.", type=int, default=1
    )
    parser.add_argument(
        "--grad_clip", help="Gradient clipping value.", type=float, default=350
    )
    parser.add_argument(
        "--grad_skip", help="Skip update grad norm threshold.", type=float, default=500
    )
    parser.add_argument(
        "--accu_steps", help="Gradient accumulation steps.", type=int, default=1
    )
    parser.add_argument(
        "--beta", help="Max KL beta penalty weight.", type=float, default=1.0
    )
    parser.add_argument(
        "--beta_warmup_steps", help="KL beta penalty warmup steps.", type=int, default=0
    )
    parser.add_argument(
        "--kl_free_bits", help="KL min free bits constraint.", type=float, default=0.0
    )
    parser.add_argument(
        "--viz_freq", help="Steps per visualisation.", type=int, default=5000
    )
    parser.add_argument(
        "--eval_freq", help="Train epochs per validation.", type=int, default=5
    )
    parser.add_argument(
        "--enc_arch",
        help="Encoder architecture config.",
        type=str,
        default="64b1d2,32b1d2,16b1d2,8b1d8,1b2",
    )
    parser.add_argument(
        "--dec_arch",
        help="Decoder architecture config.",
        type=str,
        default="1b2,8b2,16b2,32b2,64b2",
    )
    parser.add_argument(
        "--cond_prior",
        help="Use a conditional prior.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--widths",
        help="Number of channels.",
        nargs="+",
        type=int,
        default=[16, 32, 48, 64, 128],
    )
    parser.add_argument(
        "--bottleneck", help="Bottleneck width factor.", type=int, default=4
    )
    parser.add_argument(
        "--z_dim", help="Numver of latent channel dims.", type=int, default=16
    )
    parser.add_argument(
        "--z_max_res",
        help="Max resolution of stochastic z layers.",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--bias_max_res",
        help="Learned bias param max resolution.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--x_like",
        help="x likelihood: {fixed/shared/diag}_{gauss/dgauss}.",
        type=str,
        default="diag_dgauss",
    )
    parser.add_argument(
        "--std_init",
        help="Initial std for x scale. 0 is random.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--parents_x",
        help="Parents of x to condition on.",
        nargs="+",
        default=["mri_seq", "brain_volume", "ventricle_volume", "sex"],
    )
    parser.add_argument(
        "--concat_pa",
        help="Whether to concatenate parents_x.",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--context_dim",
        help="Num context variables conditioned on.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--context_norm",
        help='Conditioning normalisation {"[-1,1]"/"[0,1]"/log_standard}.',
        type=str,
        default="log_standard",
    )
    parser.add_argument(
        "--q_correction",
        help="Use posterior correction.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--cond_drop",
        help="Use counterfactual dropout",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--embd_dim",
        help="Embedding dim",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--p_dropout",
        help="Block dropout",
        type=float,
        default=0.1,
    )
    return parser
