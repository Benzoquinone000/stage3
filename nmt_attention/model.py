from __future__ import annotations

import random

import torch
from torch import nn


class Encoder(nn.Module):
    """Bidirectional GRU encoder used to produce source-side annotations."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            embedding_dim,
            encoder_hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.hidden_bridge = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(source))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            source_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=source.shape[1],
        )

        forward_final = hidden[-2]
        backward_final = hidden[-1]
        decoder_init = torch.tanh(
            self.hidden_bridge(torch.cat((forward_final, backward_final), dim=1))
        )
        return outputs, decoder_init


class BahdanauAttention(nn.Module):
    """Additive attention from Bahdanau et al. (2014)."""

    def __init__(self, encoder_hidden_dim: int, decoder_hidden_dim: int) -> None:
        super().__init__()
        self.energy = nn.Linear(
            encoder_hidden_dim * 2 + decoder_hidden_dim, decoder_hidden_dim
        )
        self.score = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        source_len = encoder_outputs.shape[1]
        repeated_hidden = decoder_hidden.unsqueeze(1).repeat(1, source_len, 1)
        energy = torch.tanh(
            self.energy(torch.cat((repeated_hidden, encoder_outputs), dim=2))
        )
        scores = self.score(energy).squeeze(2)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=1)


class ConditionalGRUCell(nn.Module):
    """GRU transition where the attention context contributes to every gate."""

    def __init__(
        self,
        embedding_dim: int,
        context_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.input_linear = nn.Linear(embedding_dim, hidden_dim * 3, bias=True)
        self.hidden_linear = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.context_linear = nn.Linear(context_dim, hidden_dim * 3, bias=False)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        embedded: torch.Tensor,
        hidden: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        input_update, input_reset, input_candidate = self.input_linear(embedded).chunk(
            3, dim=1
        )
        hidden_update, hidden_reset, hidden_candidate = self.hidden_linear(
            hidden
        ).chunk(3, dim=1)
        context_update, context_reset, context_candidate = self.context_linear(
            context
        ).chunk(3, dim=1)

        update = torch.sigmoid(input_update + hidden_update + context_update)
        reset = torch.sigmoid(input_reset + hidden_reset + context_reset)
        candidate = torch.tanh(
            input_candidate + reset * hidden_candidate + context_candidate
        )
        return update * candidate + (1.0 - update) * hidden


class MaxoutReadout(nn.Module):
    """Deep output layer used by RNNsearch-style decoders."""

    def __init__(
        self,
        embedding_dim: int,
        context_dim: int,
        hidden_dim: int,
        output_dim: int,
        maxout_dim: int,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.maxout_dim = maxout_dim
        self.pool_size = pool_size
        self.pre_output = nn.Linear(
            embedding_dim + context_dim + hidden_dim,
            maxout_dim * pool_size,
        )
        self.output = nn.Linear(maxout_dim, output_dim)

    def forward(
        self,
        embedded: torch.Tensor,
        hidden: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        readout = self.pre_output(torch.cat((embedded, hidden, context), dim=1))
        batch_size = readout.shape[0]
        readout = readout.view(batch_size, self.maxout_dim, self.pool_size)
        readout = readout.max(dim=2).values
        return self.output(readout)


class GroundHogMaxoutReadout(nn.Module):
    """GroundHog-style deep output: summed projections followed by Maxout(2)."""

    def __init__(
        self,
        embedding_dim: int,
        context_dim: int,
        hidden_dim: int,
        output_dim: int,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim % pool_size != 0:
            raise ValueError("hidden_dim must be divisible by pool_size.")
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size
        self.context_readout = nn.Linear(context_dim, hidden_dim, bias=False)
        self.hidden_readout = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.prev_word_readout = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim // pool_size, output_dim)

    def forward(
        self,
        embedded: torch.Tensor,
        hidden: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        readout = (
            self.context_readout(context)
            + self.hidden_readout(hidden)
            + self.prev_word_readout(embedded)
        )
        batch_size = readout.shape[0]
        readout = readout.view(
            batch_size,
            self.hidden_dim // self.pool_size,
            self.pool_size,
        )
        readout = readout.max(dim=2).values
        return self.output(readout)


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        embedding_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        dropout: float,
        pad_idx: int,
        attention: BahdanauAttention,
        readout: str = "maxout",
        maxout_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=pad_idx)
        context_dim = encoder_hidden_dim * 2
        self.rnn_cell = ConditionalGRUCell(
            embedding_dim=embedding_dim,
            context_dim=context_dim,
            hidden_dim=decoder_hidden_dim,
        )
        self.dropout = nn.Dropout(dropout)
        if readout == "maxout":
            self.readout = MaxoutReadout(
                embedding_dim=embedding_dim,
                context_dim=context_dim,
                hidden_dim=decoder_hidden_dim,
                output_dim=output_dim,
                maxout_dim=maxout_dim or decoder_hidden_dim,
            )
        elif readout == "groundhog":
            self.readout = GroundHogMaxoutReadout(
                embedding_dim=embedding_dim,
                context_dim=context_dim,
                hidden_dim=decoder_hidden_dim,
                output_dim=output_dim,
            )
        elif readout == "linear":
            self.readout = nn.Linear(
                context_dim + decoder_hidden_dim + embedding_dim, output_dim
            )
        else:
            raise ValueError(f"Unsupported readout: {readout}")

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(input_token))

        attention_weights = self.attention(hidden, encoder_outputs, mask)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        hidden = self.rnn_cell(embedded, hidden, context)
        if isinstance(self.readout, (MaxoutReadout, GroundHogMaxoutReadout)):
            prediction = self.readout(embedded, hidden, context)
        else:
            prediction = self.readout(torch.cat((hidden, context, embedded), dim=1))
        return prediction, hidden, attention_weights


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        source_pad_idx: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_pad_idx = source_pad_idx
        self.device = device

    def create_mask(self, source: torch.Tensor) -> torch.Tensor:
        return source != self.source_pad_idx

    def forward(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
        target: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, target_len = target.shape
        target_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(
            batch_size, target_len, target_vocab_size, device=self.device
        )
        attentions = torch.zeros(
            batch_size, target_len, source.shape[1], device=self.device
        )

        encoder_outputs, hidden = self.encoder(source, source_lengths)
        mask = self.create_mask(source)
        input_token = target[:, 0]

        for timestep in range(1, target_len):
            output, hidden, attention = self.decoder(
                input_token, hidden, encoder_outputs, mask
            )
            outputs[:, timestep] = output
            attentions[:, timestep] = attention

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = target[:, timestep] if teacher_force else top1

        return outputs, attentions

    @torch.no_grad()
    def translate(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 50,
    ) -> tuple[list[int], torch.Tensor]:
        self.eval()
        encoder_outputs, hidden = self.encoder(source, source_lengths)
        mask = self.create_mask(source)
        input_token = torch.tensor([sos_idx], dtype=torch.long, device=self.device)

        generated: list[int] = []
        attention_history: list[torch.Tensor] = []
        for _ in range(max_len):
            output, hidden, attention = self.decoder(
                input_token, hidden, encoder_outputs, mask
            )
            top1 = int(output.argmax(1).item())
            if top1 == eos_idx:
                break
            generated.append(top1)
            attention_history.append(attention.squeeze(0).cpu())
            input_token = torch.tensor([top1], dtype=torch.long, device=self.device)

        if attention_history:
            attentions = torch.stack(attention_history, dim=0)
        else:
            attentions = torch.empty(0, source.shape[1])
        return generated, attentions

    @torch.no_grad()
    def beam_search(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        unk_idx: int | None = None,
        max_len: int = 50,
        beam_size: int = 5,
        length_penalty: float = 0.0,
        min_len: int = 0,
        no_repeat_ngram_size: int = 0,
    ) -> tuple[list[int], torch.Tensor]:
        self.eval()
        encoder_outputs, init_hidden = self.encoder(source, source_lengths)
        mask = self.create_mask(source)

        beams = [
            {
                "tokens": [],
                "log_prob": 0.0,
                "hidden": init_hidden,
                "attentions": [],
                "finished": False,
            }
        ]

        def normalized_score(beam: dict[str, object]) -> float:
            length = max(1, len(beam["tokens"]))
            if length_penalty <= 0:
                return float(beam["log_prob"])
            penalty = ((5.0 + length) / 6.0) ** length_penalty
            return float(beam["log_prob"]) / penalty

        for _ in range(max_len):
            candidates = []
            for beam in beams:
                if beam["finished"]:
                    candidates.append(beam)
                    continue

                previous_token = beam["tokens"][-1] if beam["tokens"] else sos_idx
                input_token = torch.tensor(
                    [previous_token], dtype=torch.long, device=self.device
                )
                output, next_hidden, attention = self.decoder(
                    input_token,
                    beam["hidden"],
                    encoder_outputs,
                    mask,
                )
                log_probs = torch.log_softmax(output, dim=1)
                if unk_idx is not None:
                    log_probs[:, unk_idx] = -torch.inf
                if len(beam["tokens"]) < min_len:
                    log_probs[:, eos_idx] = -torch.inf
                if no_repeat_ngram_size > 0:
                    banned_tokens = self._get_banned_tokens(
                        list(beam["tokens"]), no_repeat_ngram_size
                    )
                    if banned_tokens:
                        log_probs[:, banned_tokens] = -torch.inf
                top_log_probs, top_indices = log_probs.topk(beam_size, dim=1)

                for log_prob, token_idx in zip(top_log_probs[0], top_indices[0]):
                    token = int(token_idx.item())
                    new_tokens = list(beam["tokens"])
                    finished = token == eos_idx
                    if not finished:
                        new_tokens.append(token)

                    candidates.append(
                        {
                            "tokens": new_tokens,
                            "log_prob": float(beam["log_prob"])
                            + float(log_prob.item()),
                            "hidden": next_hidden.clone(),
                            "attentions": list(beam["attentions"])
                            + [attention.squeeze(0).cpu()],
                            "finished": finished,
                        }
                    )

            beams = sorted(candidates, key=normalized_score, reverse=True)[:beam_size]
            if all(beam["finished"] for beam in beams):
                break

        best = max(beams, key=normalized_score)
        selected_attentions = best["attentions"][: len(best["tokens"])]
        if selected_attentions:
            attentions = torch.stack(selected_attentions, dim=0)
        else:
            attentions = torch.empty(0, source.shape[1])
        return list(best["tokens"]), attentions

    @staticmethod
    def _get_banned_tokens(tokens: list[int], ngram_size: int) -> list[int]:
        if ngram_size <= 0 or len(tokens) + 1 < ngram_size:
            return []
        prefix = tuple(tokens[-(ngram_size - 1) :]) if ngram_size > 1 else tuple()
        banned: list[int] = []
        for start in range(0, len(tokens) - ngram_size + 1):
            ngram = tuple(tokens[start : start + ngram_size])
            if ngram_size == 1 or ngram[:-1] == prefix:
                banned.append(ngram[-1])
        return banned


def build_model(
    source_vocab_size: int,
    target_vocab_size: int,
    source_pad_idx: int,
    target_pad_idx: int,
    device: torch.device,
    embedding_dim: int = 128,
    encoder_hidden_dim: int = 256,
    decoder_hidden_dim: int = 256,
    dropout: float = 0.2,
    readout: str = "maxout",
    maxout_dim: int | None = None,
    init: str = "paper",
) -> Seq2Seq:
    attention = BahdanauAttention(encoder_hidden_dim, decoder_hidden_dim)
    encoder = Encoder(
        input_dim=source_vocab_size,
        embedding_dim=embedding_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        dropout=dropout,
        pad_idx=source_pad_idx,
    )
    decoder = Decoder(
        output_dim=target_vocab_size,
        embedding_dim=embedding_dim,
        encoder_hidden_dim=encoder_hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        dropout=dropout,
        pad_idx=target_pad_idx,
        attention=attention,
        readout=readout,
        maxout_dim=maxout_dim,
    )
    model = Seq2Seq(encoder, decoder, source_pad_idx, device).to(device)
    if init == "paper":
        init_paper_style(model, source_pad_idx, target_pad_idx)
    elif init != "default":
        raise ValueError(f"Unsupported init mode: {init}")
    return model


def init_paper_style(
    model: nn.Module, source_pad_idx: int, target_pad_idx: int
) -> None:
    """Initialize weights close to the GroundHog RNNsearch recipe."""

    for module in model.modules():
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRU):
            for name, parameter in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.normal_(parameter, mean=0.0, std=0.01)
                elif "weight_hh" in name:
                    for chunk in parameter.chunk(3, dim=0):
                        nn.init.orthogonal_(chunk)
                elif "bias" in name:
                    nn.init.zeros_(parameter)

    for module in model.modules():
        if isinstance(module, ConditionalGRUCell):
            hidden_chunks = module.hidden_linear.weight.chunk(3, dim=0)
            with torch.no_grad():
                for chunk in hidden_chunks:
                    nn.init.orthogonal_(chunk)

    if hasattr(model.decoder.attention, "score"):
        nn.init.zeros_(model.decoder.attention.score.weight)

    for pad_idx, embedding in (
        (source_pad_idx, model.encoder.embedding),
        (target_pad_idx, model.decoder.embedding),
    ):
        with torch.no_grad():
            embedding.weight[pad_idx].fill_(0.0)
