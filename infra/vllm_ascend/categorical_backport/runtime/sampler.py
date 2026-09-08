import os

import torch
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.triton_utils import HAS_TRITON
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.sample.penalties import apply_all_penalties
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, global_stream, npu_stream_switch

DEFAULT_LOGPROBS_MODE = "raw_logprobs"
_CATEGORICAL_SAMPLE_ENABLED = os.getenv("VLLM_ASCEND_ENABLE_CATEGORICAL_SAMPLE", "0") == "1"
_NPU_GENERATOR_COMPATIBILITY_INCREMENT = 12
_CATEGORICAL_SAMPLE_ACTIVATED = False


def native_categorical_sample(
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor | None:
    """Use the stateless AscendC sampler for fully request-seeded batches."""
    global _CATEGORICAL_SAMPLE_ACTIVATED
    rows = logits.shape[0]
    if (
        not _CATEGORICAL_SAMPLE_ENABLED
        or len(generators) != rows
        or set(generators) != set(range(rows))
        or not hasattr(torch.ops._C_ascend, "npu_categorical_sample")
    ):
        return None

    ordered = [generators[index] for index in range(rows)]
    seeds = [int(generator.initial_seed()) for generator in ordered]
    offsets = [int(generator.get_offset()) for generator in ordered]
    if any(seed < 0 or seed >= 2**63 for seed in seeds):
        return None
    if any(offset < 0 or offset % _NPU_GENERATOR_COMPATIBILITY_INCREMENT for offset in offsets):
        return None

    device = logits.device
    mapping = torch.arange(rows, dtype=torch.int32, device=device)
    temperature = torch.ones(rows, dtype=torch.float32, device=device)
    seed = torch.tensor(seeds, dtype=torch.int64, device=device)
    position = torch.tensor(
        [offset // _NPU_GENERATOR_COMPATIBILITY_INCREMENT for offset in offsets],
        dtype=torch.int64,
        device=device,
    )
    sampled, _ = torch.ops._C_ascend.npu_categorical_sample(
        logits,
        mapping,
        temperature,
        seed,
        position,
        False,
        False,
        None,
        None,
        False,
    )
    if not _CATEGORICAL_SAMPLE_ACTIVATED:
        print("vLLM-Ascend native categorical sampler activated", flush=True)
        _CATEGORICAL_SAMPLE_ACTIVATED = True
    for generator, offset in zip(ordered, offsets, strict=True):
        generator.set_offset(offset + _NPU_GENERATOR_COMPATIBILITY_INCREMENT)
    return sampled.view(-1)


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-NPU synchronization.
    """
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    with npu_stream_switch(global_stream()):
        q = torch.empty_like(probs)
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            # TODO(woosuk): This can be slow because we handle each request
            # one by one. Optimize this.
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.current_stream().wait_stream(global_stream())
    return probs.div_(q).argmax(dim=-1).view(-1)


class AscendSampler(Sampler):
    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        """Use Triton-Ascend penalties on NPU when Triton is available; else vLLM default."""
        if not HAS_TRITON:
            return Sampler.apply_penalties(logits, sampling_metadata, output_token_ids)

        if sampling_metadata.no_penalties:
            return logits
        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )

    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        # TODO: support logprobs_mode in vllm-ascend
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler()
        self.async_exponential_event = torch.npu.Event()

    def set_q_event(self, q, event):
        self.topk_topp_sampler.set_q_event(q, event)

    def do_async_exponential(self, b_s, head_dim, generators):
        # Calculating exponential randoms in a different stream
        # and overlapping with model executing.
        with torch.npu.stream(global_stream()):
            global_stream().wait_stream(torch.npu.current_stream())
            q = torch.empty((b_s, head_dim), device="npu", dtype=torch.float32)
            # Goes to async exponential with AI-CPU exponential or default exponential.
            if len(generators) != q.shape[0]:
                q.exponential_()
            if generators:
                for i, generator in generators.items():
                    q[i].exponential_(generator=generator)
            self.async_exponential_event.record()
        self.set_q_event(q, self.async_exponential_event)


class AscendTopKTopPSampler(TopKTopPSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_top_k_top_p = apply_top_k_top_p

    def set_q_event(self, q, event):
        # Pass in async exponential results.
        # Also pass in event to prevent synchronize errors.
        self.q = q
        self.async_event = event

    def forward_native(self, logits, generators, k, p):
        """Override pytorch native implementation to torch_npu"""
        # when batch_invariant mode is enabled, we should use vllm's implementation.
        # or it will make batch_invariant mode not working.
        if vllm_is_batch_invariant():
            return super().forward_native(logits, generators, k, p)
        logits = self.apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        sampled = native_categorical_sample(logits, generators)
        if sampled is not None:
            return sampled, logits_to_return

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        if get_ascend_config().enable_async_exponential:
            # Add synchronize to prevent synchronize error.
            self.async_event.synchronize()
            return probs.div_(self.q).argmax(dim=-1).view(-1), logits_to_return
        return random_sample(probs, generators), logits_to_return


def _apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    probs = logits.softmax(dim=-1)
    probs_sort, _ = probs.sort(dim=-1, descending=False)

    if k is not None:
        top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
        top_k_count = top_k_count.unsqueeze(dim=1)
        top_k_cutoff = probs_sort.gather(-1, top_k_count)

        # Make sure the no top-k rows are no-op.
        no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
        top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

        elements_to_discard = probs < top_k_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    if p is not None:
        cumprob = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one

        top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
        top_p_cutoff = probs_sort.gather(-1, top_p_count)
        elements_to_discard = probs < top_p_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    return logits


def _apply_top_k_top_p_ascendc(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    return torch.ops._C_ascend.npu_apply_top_k_top_p(logits, k=k, p=p)


apply_top_k_top_p = (
    _apply_top_k_top_p_ascendc
    if get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]
    else _apply_top_k_top_p_pytorch
)
