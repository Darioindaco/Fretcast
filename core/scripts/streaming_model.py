"""Fase 3 shared pieces: unidirectional + limited-lookahead model construction and labels.

Plan section 1's "verplichte ingreep": the bidirectional GRU that fase 1/2 trained
cannot run causally frame-by-frame -- it needs the whole sequence, including frames
that haven't happened yet in a live performance. A unidirectional (causal) GRU can
stream, but loses the small amount of future context that helps disambiguate an onset.
The fix is a fixed, small lookahead: the model is trained to predict frame t's label
using audio up to frame (t + LOOKAHEAD_FRAMES) -- at inference this costs exactly
LOOKAHEAD_FRAMES worth of latency (buffer that many extra frames before emitting a
decision), which is the plan's explicit trade: ~50-70ms of lookahead in exchange for a
causal, streamable model.

LOOKAHEAD_FRAMES=3 at hop_length=512/sample_rate=22050 (trimplexx's own config) is
3*512/22050 = 69.7ms, landing inside the plan's stated 50-70ms budget.
"""
from __future__ import annotations

import sys
from pathlib import Path

TRIMPLEXX_PY = Path(__file__).resolve().parents[2] / "external" / "trimplexx" / "python"
sys.path.insert(0, str(TRIMPLEXX_PY))

import torch  # noqa: E402
import config  # noqa: E402
from data_processing import dataset  # noqa: E402
from model import architecture  # noqa: E402

LOOKAHEAD_FRAMES = 3
LOOKAHEAD_SECONDS = LOOKAHEAD_FRAMES * config.HOP_LENGTH / config.SAMPLE_RATE


def create_frame_level_labels_with_lookahead(
    raw_annotation_tensor,
    feature_map_tensor,
    frame_hop_length,
    audio_sr,
    fret_max_value,
    lookahead_frames: int = LOOKAHEAD_FRAMES,
):
    """Same target tensors as trimplexx's own create_frame_level_labels, shifted so that
    position [t] holds the label that was really at [t - lookahead_frames]. A causal
    model reading up to frame t is being asked for a frame that's already
    lookahead_frames in its past, i.e. it's allowed to "see" that many frames past the
    event before committing -- the lookahead itself, encoded directly into training.
    """
    onset_targets, fret_targets = dataset.create_frame_level_labels(
        raw_annotation_tensor, feature_map_tensor, frame_hop_length, audio_sr, fret_max_value
    )
    if lookahead_frames <= 0:
        return onset_targets, fret_targets

    num_frames = onset_targets.shape[0]
    silence_fret_idx = fret_max_value + config.FRET_SILENCE_CLASS_OFFSET

    shifted_onset = torch.zeros_like(onset_targets)
    shifted_fret = torch.full_like(fret_targets, silence_fret_idx)
    if lookahead_frames < num_frames:
        shifted_onset[lookahead_frames:] = onset_targets[: num_frames - lookahead_frames]
        shifted_fret[lookahead_frames:] = fret_targets[: num_frames - lookahead_frames]
    return shifted_onset, shifted_fret


def build_unidirectional_model_warm_started(bidirectional_checkpoint_path, bidirectional_run_config, device):
    """Construct a fresh unidirectional GuitarTabCRNN, warm-starting its CNN front end
    from an already-trained (bidirectional) checkpoint. Only the CNN transfers cleanly:
    `rnn_bidirectional` changes the RNN's own weight shapes (separate fwd/bwd matrices)
    and doubles onset_fc/fret_fc's input width (rnn_output_size), so those layers must
    train from scratch -- the CNN (TabCNN) never depends on rnn_bidirectional at all, so
    its weights are architecture-compatible as-is.
    """
    hyperparams = bidirectional_run_config["hyperparameters_tuned"]
    temp_cnn = architecture.TabCNN()
    with torch.no_grad():
        cnn_out = temp_cnn(torch.randn(1, 1, config.N_BINS_CQT, 32))
        rnn_input_dim = cnn_out.shape[1] * cnn_out.shape[2]
    del temp_cnn

    model = architecture.GuitarTabCRNN(
        num_frames_rnn_input_dim=rnn_input_dim,
        rnn_type=hyperparams.get("RNN_TYPE", config.RNN_TYPE_DEFAULT),
        rnn_hidden_size=hyperparams["RNN_HIDDEN_SIZE"],
        rnn_layers=hyperparams["RNN_LAYERS"],
        rnn_dropout=hyperparams["RNN_DROPOUT"],
        rnn_bidirectional=False,
    )

    checkpoint_state = torch.load(str(bidirectional_checkpoint_path), map_location=device, weights_only=True)
    cnn_only_state = {k: v for k, v in checkpoint_state.items() if k.startswith("cnn.")}
    missing, unexpected = model.load_state_dict(cnn_only_state, strict=False)
    still_missing_non_cnn = [k for k in missing if k.startswith("cnn.")]
    if still_missing_non_cnn:
        raise RuntimeError(f"CNN weights failed to transfer: {still_missing_non_cnn}")

    model.to(device)
    return model
