# ruff: noqa: E501
from __future__ import annotations

# ------------------------------------------------------------------ the model snapshot (weights)
# The pinned snapshot is the authors' Hugging Face Space, which hosts the paper checkpoint byte-for-byte: its
# `checkpoint_best_regular.pth` has the SHA-256 of the file the upstream README links on Google Drive
# (`checkpoint_fsc147_best.pth`, 1.2 GB; both downloaded and hashed on 2026-09-24). The checkpoint is a pickle,
# so it is audited and converted once into the safetensors file the package serves (see docs/WEIGHTS.md).
MODEL_ID = "nikigoli/countgd"
MODEL_REPO_TYPE = "space"
MODEL_REVISION = "6e82e59569a84ee5c6aafa35d396f2d2bee57be2"
MODEL_LICENSE = "MIT"
DEFAULT_MODEL_KEY = "countgd"
SOURCE_CKPT_NAME = "checkpoint_best_regular.pth"
SOURCE_CKPT_SHA256 = "c1bab864b17db345b4c6e3aaabb5765bc2c0a90d0bc8defb5e664a74a50aa126"
SOURCE_CKPT_SIZE_BYTES = 1_250_122_522
SOURCE_DRIVE_ID = "1RbRcNLsOfeEbx6u39pBehqsgQiexHHrI"  # the upstream README's link; same bytes as the Space file
ALLOWED_CHECKPOINT_FILES = ("README.md", SOURCE_CKPT_NAME)  # the manifest-listed files of the Space snapshot
# Every GLOBAL the checkpoint's pickle names (static `pickletools` audit; nothing executed). `argparse.Namespace`
# is the training-arguments object; it is the one class beyond torch's default restricted-unpickler set.
PICKLE_ALLOWED_GLOBALS = (
    "argparse.Namespace",
    "collections.OrderedDict",
    "torch.FloatStorage",
    "torch.LongStorage",
    "torch._utils._rebuild_tensor_v2",
)
PICKLE_AUDIT_SHA256 = "4606eaf365d27d2fddd901bdc069218dabbd418f9856ed2d0e3707612ad4c527"  # sha256 of the sorted names, newline-joined
SOURCE_STATE_KEY = "model"  # the checkpoint also carries args, epoch (19), lr_scheduler and optimizer, which are not read
SOURCE_STATE_TENSORS = 1_146
DROPPED_PREFIX = "feature_map_encoder."  # 38 tensors of an exemplar encoder the forward pass never calls
# The converted serving file: the model's state dict with each tied box-head tensor stored once and the 66
# aliases recorded as one JSON metadata entry (safetensors writes a multi-key metadata map in hash order, so
# only a single key keeps the file's digest reproducible).
MODEL_FILENAME = "countgd.safetensors"
MODEL_SHA256 = "8e44867b951e3a4205d918e022b78bc5fea218fd17c1851b864a01c421d2d443"
MODEL_SIZE_BYTES = 937_560_480
ALIAS_METADATA_KEY = "countgd_aliases"
# The upstream code the vendored `modeling.py` is carried from.
UPSTREAM_REPOSITORY = "niki-amini-naieni/CountGD"
UPSTREAM_COMMIT = "b6f362b3f5cd20db4a171faa410dfed8f2f466d8"
UPSTREAM_LICENSE = "MIT"  # the GroundingDINO parts carry IDEA's Apache-2.0 header

# ------------------------------------------------------------------ the text-tokenizer snapshot (no weights)
TOKENIZER_MODEL_ID = "google-bert/bert-base-uncased"
TOKENIZER_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
TOKENIZER_LICENSE = "Apache-2.0"
TOKENIZER_KEY = "bert-base-uncased"
TOKENIZER_FILES = ("LICENSE", "README.md", "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt")
# BERT's weights are *inside* the CountGD checkpoint (`bert.*`, 199 tensors); only its vocabulary, tokenizer
# configuration and architecture configuration come from this snapshot.

# ------------------------------------------------------------------ architecture (config/cfg_fsc147_vit_b_test.py upstream)
MODEL_CONFIG: dict[str, object] = {
    "modelname": "groundingdino",
    "backbone": "swin_B_384_22k",
    "position_embedding": "sine",
    "pe_temperatureH": 20,
    "pe_temperatureW": 20,
    "return_interm_indices": [1, 2, 3],
    "backbone_freeze_keywords": None,
    "enc_layers": 6,
    "dec_layers": 6,
    "pre_norm": False,
    "dim_feedforward": 2048,
    "hidden_dim": 256,
    "dropout": 0.0,
    "nheads": 8,
    "num_queries": 900,
    "query_dim": 4,
    "num_patterns": 0,
    "num_feature_levels": 4,
    "enc_n_points": 4,
    "dec_n_points": 4,
    "two_stage_type": "standard",
    "two_stage_bbox_embed_share": False,
    "two_stage_class_embed_share": False,
    "transformer_activation": "relu",
    "dec_pred_bbox_embed_share": True,
    "dn_box_noise_scale": 1.0,
    "dn_label_noise_ratio": 0.5,
    "dn_labelbook_size": 91,
    "embed_init_tgt": True,
    "max_text_len": 256,
    "text_encoder_type": "bert-base-uncased",
    "use_text_enhancer": True,
    "use_fusion_layer": True,
    "use_checkpoint": False,  # upstream trains with activation checkpointing on; inference and the bounded adaptation here do not need it
    "use_transformer_ckpt": False,
    "use_text_cross_attention": True,
    "text_dropout": 0.0,
    "fusion_dropout": 0.0,
    "fusion_droppath": 0.1,
    "sub_sentence_present": True,
    "aux_loss": True,
    "set_cost_class": 5.0,
    "set_cost_bbox": 1.0,
    "set_cost_giou": 0.0,
    "cls_loss_coef": 5.0,
    "bbox_loss_coef": 1.0,
    "giou_loss_coef": 0.0,
    "interm_loss_coef": 1.0,
    "no_interm_box_loss": False,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "matcher_type": "HungarianMatcher",
}
PARAMETER_COUNT = 233_362_816  # parameters of the built model (232,772,224 of them require grad at construction; the checkpoint adds 24 int64 index buffers)
STATE_TENSORS = 1_108  # state-dict entries, 66 of them aliases of the shared box head recorded in the safetensors metadata
HIDDEN_DIM = 256
NUM_QUERIES = 900
DECODER_LAYERS = 6
ENCODER_LAYERS = 6

# ------------------------------------------------------------------ inference protocol (upstream inference scripts)
CONFIDENCE_THRESHOLD = 0.23  # upstream `--confidence_thresh` / cfg `box_threshold`: a query counts when its best token score exceeds it
SHORT_SIDE = 800  # upstream test transform: RandomResize([800], max_size=1333)
MAX_SIDE = 1333
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
MAX_EXEMPLARS = 3  # FSC-147 gives three exemplar boxes per image; upstream uses all three
TARGET_BOX_PX = 2.0  # upstream's FSC-147 training boxes: a 2 x 2 px box centred on each annotated point
EXEMPLAR_TOKEN_ID = 1008  # the `*` WordPiece id upstream inserts for each visual exemplar
MAX_TEXT_CHARS = 64

UNSAFE_WEIGHT_EXTENSIONS = (  # refused anywhere except the audited source checkpoint itself
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".h5",
    ".msgpack",
)
