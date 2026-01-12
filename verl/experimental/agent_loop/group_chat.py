# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import os
import re
from typing import Any
from uuid import uuid4

import numpy as np
import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.ray_utils import get_event_loop

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _build_contiguous_groups(group_ids: np.ndarray) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    last_id = None
    for i, gid in enumerate(group_ids.tolist()):
        if last_id is None or gid == last_id:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
        last_id = gid
    if current:
        groups.append(current)
    return groups


def _split_groups(groups: list[list[int]], num_splits: int) -> list[list[list[int]]]:
    if num_splits <= 0:
        return [groups]
    if not groups:
        return [[] for _ in range(num_splits)]
    idx_chunks = np.array_split(np.arange(len(groups)), num_splits)
    return [[groups[i] for i in chunk.tolist()] for chunk in idx_chunks]


class GroupChatAgentLoopWorker(AgentLoopWorker):
    """Group-level multi-round rollout with shared context across rollouts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_config = self.config.data
        self.apply_chat_template_kwargs = self.dataset_config.get("apply_chat_template_kwargs", {})
        self.system_prompt = initialize_system_prompt(self.tokenizer, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    def _get_group_cfg(self) -> dict[str, Any]:
        rollout_cfg = self.config.actor_rollout_ref.rollout
        custom = rollout_cfg.get("custom", {}) or {}
        return custom.get("group_chat", {}) or {}

    async def _process_vision_info(self, messages: list[dict]) -> dict:
        multi_modal_data = {}
        if self.processor is not None:
            images, videos = await self.dataset_cls.process_vision_info(
                messages, image_patch_size=self.processor.image_processor.patch_size, config=self.dataset_config
            )
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos
        return multi_modal_data

    async def _apply_chat_template(
        self,
        messages: list[dict],
        *,
        images=None,
        videos=None,
        remove_system_prompt: bool = False,
    ) -> list[int]:
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=None,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            if videos is not None:
                videos, video_metadatas = zip(*videos, strict=False)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None
            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                video_metadatas=video_metadatas,
                return_tensors="pt",
                do_sample_frames=False,
            )
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=None,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]
        return prompt_ids

    async def _run_group(
        self,
        indices: list[int],
        kwargs_list: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        group_cfg: dict[str, Any],
    ) -> dict[int, AgentLoopOutput]:
        num_rounds = int(group_cfg.get("num_rounds", 2))
        strip_think = bool(group_cfg.get("strip_think", True))
        peer_role = group_cfg.get("peer_role", "assistant")

        states = []
        for idx in indices:
            kwargs = kwargs_list[idx]
            messages = list(kwargs["raw_prompt"])
            multi_modal_data = await self._process_vision_info(messages)
            images = multi_modal_data.get("images")
            videos = multi_modal_data.get("videos")
            prompt_ids = await self._apply_chat_template(messages, images=images, videos=videos)
            states.append(
                {
                    "request_id": uuid4().hex,
                    "prompt_ids": prompt_ids,
                    "response_mask": [],
                    "response_logprobs": [] if sampling_params.get("logprobs") else None,
                    "multi_modal_data": multi_modal_data,
                    "images": images,
                    "videos": videos,
                }
            )

        for round_idx in range(num_rounds):
            tasks = []
            for state in states:
                tasks.append(
                    self.server_manager.generate(
                        request_id=state["request_id"],
                        prompt_ids=state["prompt_ids"],
                        sampling_params=sampling_params,
                        image_data=state["images"],
                        video_data=state["videos"],
                    )
                )
            outputs = await asyncio.gather(*tasks)

            sanitized_texts = []
            for state, output in zip(states, outputs, strict=True):
                token_ids = output.token_ids
                state["prompt_ids"] += token_ids
                state["response_mask"] += [1] * len(token_ids)
                if state["response_logprobs"] is not None:
                    if output.log_probs:
                        state["response_logprobs"] += output.log_probs
                    else:
                        state["response_logprobs"] += [0.0] * len(token_ids)
                raw = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                sanitized_texts.append(_strip_think(raw) if strip_think else raw)

            if round_idx == num_rounds - 1:
                break

            for i, state in enumerate(states):
                add_messages = []
                for j, text in enumerate(sanitized_texts):
                    if j == i:
                        continue
                    if not text:
                        continue
                    add_messages.append({"role": peer_role, "content": text})
                if not add_messages:
                    continue
                peer_ids = await self._apply_chat_template(add_messages, remove_system_prompt=True)
                state["prompt_ids"] += peer_ids
                state["response_mask"] += [0] * len(peer_ids)
                if state["response_logprobs"] is not None:
                    state["response_logprobs"] += [0.0] * len(peer_ids)

        response_length = self.config.actor_rollout_ref.rollout.response_length
        outputs_by_idx: dict[int, AgentLoopOutput] = {}
        for idx, state in zip(indices, states, strict=True):
            if state["response_mask"]:
                response_ids = state["prompt_ids"][-len(state["response_mask"]) :]
                prompt_ids = state["prompt_ids"][: len(state["prompt_ids"]) - len(state["response_mask"])]
            else:
                response_ids = []
                prompt_ids = state["prompt_ids"]

            response_mask = state["response_mask"][:response_length]
            response_ids = response_ids[:response_length]
            response_logprobs = None
            if state["response_logprobs"] is not None:
                response_logprobs = state["response_logprobs"][:response_length]

            outputs_by_idx[idx] = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                response_logprobs=response_logprobs,
                multi_modal_data=state["multi_modal_data"],
                num_turns=num_rounds + 1,
                metrics=AgentLoopMetrics(),
                extra_fields={},
            )
        return outputs_by_idx

    async def generate_sequences(self, batch):
        group_cfg = self._get_group_cfg()
        if not group_cfg.get("enable", True):
            return await super().generate_sequences(batch)

        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "uid" in batch.non_tensor_batch:
            group_ids = batch.non_tensor_batch["uid"]
        elif "index" in batch.non_tensor_batch:
            group_ids = batch.non_tensor_batch["index"]
        else:
            group_ids = np.arange(len(batch))

        kwargs_list = [{k: v[i] for k, v in batch.non_tensor_batch.items()} for i in range(len(batch))]
        groups = _build_contiguous_groups(group_ids)

        tasks = [
            asyncio.create_task(self._run_group(indices, kwargs_list, sampling_params, group_cfg))
            for indices in groups
        ]
        group_outputs = await asyncio.gather(*tasks)

        outputs: list[AgentLoopOutput] = [None] * len(batch)
        for group_output in group_outputs:
            for idx, output in group_output.items():
                outputs[idx] = output

        post_tasks = [
            asyncio.create_task(self._agent_loop_postprocess(outputs[i], **kwargs_list[i]))
            for i in range(len(outputs))
        ]
        internal_outputs = await asyncio.gather(*post_tasks)
        return self._postprocess(internal_outputs)


class GroupChatAgentLoopManager(AgentLoopManager):
    """AgentLoopManager that keeps rollout groups intact before dispatching to workers."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(GroupChatAgentLoopWorker)
        super().__init__(*args, **kwargs)

    def _get_group_cfg(self) -> dict[str, Any]:
        rollout_cfg = self.config.actor_rollout_ref.rollout
        custom = rollout_cfg.get("custom", {}) or {}
        return custom.get("group_chat", {}) or {}

    def generate_sequences(self, prompts):
        group_cfg = self._get_group_cfg()
        if not group_cfg.get("enable", True):
            return super().generate_sequences(prompts)

        self.wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()

        if "uid" in prompts.non_tensor_batch:
            group_ids = prompts.non_tensor_batch["uid"]
        elif "index" in prompts.non_tensor_batch:
            group_ids = prompts.non_tensor_batch["index"]
        else:
            return super().generate_sequences(prompts)

        groups = _build_contiguous_groups(group_ids)
        group_chunks = _split_groups(groups, len(self.agent_loop_workers))

        chunks = []
        workers = []
        for worker, group_chunk in zip(self.agent_loop_workers, group_chunks, strict=False):
            if not group_chunk:
                continue
            indices = [idx for group in group_chunk for idx in group]
            chunks.append(prompts.select_idxs(indices))
            workers.append(worker)

        outputs = ray.get([worker.generate_sequences.remote(chunk) for worker, chunk in zip(workers, chunks)])
        output = prompts.__class__.concat(outputs)

        self.sleep()
        if self.reward_model_manager:
            self.reward_model_manager.sleep()

        metrics = [output_item.meta_info.pop("metrics") for output_item in outputs]
        timing = self._performance_metrics(metrics, output)
        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output
