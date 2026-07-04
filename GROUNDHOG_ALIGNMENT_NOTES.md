# GroundHog Alignment Notes

GroundHog is the closest public reference implementation for Bahdanau et al.'s
RNNsearch experiments. The project-local files `reference_groundhog_state.py`,
`reference_groundhog_encdec.py`, and `reference_groundhog_train.py` contain the
key NMT configuration/model snippets used for this comparison.

## Exact RNNsearch-50 State

`prototype_search_state()` is explicitly documented as the configuration used to
train the paper's RNNsearch-50 model.

Important values:

- `search = True`
- `dec_rec_layer = RecurrentLayerWithSearch`
- `forward = True`
- `backward = True`
- `last_forward = False`
- `seqlen = 50`
- `bs = 80`
- `sort_k_batches = 20`
- `dim = 1000`
- `rank_n_approx = 620`
- `algo = SGD_adadelta`
- `adarho = 0.95`
- `adaeps = 1e-6`
- `cutoff = 1.0`
- `weight_scale = 0.01`
- `rec_weight_init_fn = sample_weights_orth`
- `deep_out = True`
- `unary_activ = Maxout(2)`

The current PyTorch strict preset matches the major numeric hyperparameters.

## Data/Vocabulary Details

GroundHog points to already-selected and binarized data:

- `vocab.unlimited/bitexts.selected/binarized_text.shuffled.en.h5`
- `vocab.unlimited/bitexts.selected/binarized_text.shuffled.fr.h5`
- vocabulary pickles from the same selected bitext directory

This confirms that the final paper-style experiment should not be run on raw
WMT14 directly. It should run on the Axelrod-selected subset, then binarize/map
tokens to the paper vocabulary.

Special symbol layout in GroundHog:

- `unk_sym_source = 1`
- `unk_sym_target = 1`
- `null_sym_source = 30000`
- `null_sym_target = 30000`
- `n_sym_source = 30001`
- `n_sym_target = 30001`

The PyTorch implementation currently uses explicit `<pad>`, `<sos>`, `<eos>`,
and `<unk>` symbols, so its output vocabulary is not index-identical to
GroundHog. This is acceptable for a faithful PyTorch reimplementation, but a
maximally strict reproduction should add a GroundHog-compatible vocabulary mode.

## Model Structure Check

Confirmed matches:

- Bidirectional recurrent encoder.
- Source annotations concatenate forward and backward states, so context
  dimension is `2 * dim`.
- Additive attention scores each source annotation from the previous decoder
  state.
- Attention score vector is initialized to zero in GroundHog; the PyTorch
  implementation mirrors this by zeroing the attention scoring weight.
- The attention context contributes to candidate, reset gate, and update gate in
  the decoder transition.
- Output layer uses a deep/maxout readout.

Important remaining implementation gaps:

- GroundHog's readout is a sum of separate projections from context, decoder
  hidden state, and previous target embedding, followed by `Maxout(2)`. The
  PyTorch implementation now exposes `--readout groundhog` for this layout; the
  older concat-based readout remains available as `--readout maxout` for
  checkpoint compatibility.
- GroundHog's target-side training convention uses `null_sym = 30000` as EOS and
  starts generation from word id `0`; the PyTorch implementation uses a separate
  `<sos>` token.
- GroundHog's softmax layer receives `rank_n_approx = 620`; the exact internal
  low-rank/factorized behavior should be checked before claiming bit-level
  architectural equivalence.

## Training Loop Check

Confirmed matches:

- Adadelta optimizer.
- Gradient clipping threshold `1.0`.
- Minibatch size `80`.
- Homogeneous batching by sorting `sort_k_batches * batch_size` examples by
  max(source length, target length).
- Training max length `50` for RNNsearch-50.

The PyTorch implementation now keeps dev/test unfiltered when fixed official
validation/test TSVs are supplied, which matches the paper evaluation setup
better than filtering them by the training length.

## Next Fixes

1. Rebuild or obtain the 348M-word Axelrod-selected WMT14 EN-FR corpus.
2. Add an optional GroundHog-compatible vocabulary/indexing mode.
3. Verify the output softmax parameterization against GroundHog's `SoftmaxLayer`.
4. Run RNNsearch-50 with the selected corpus, then evaluate on full
   `newstest2014`.
