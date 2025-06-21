from math_verify import parse, verify
from openai import AsyncOpenAI

from verl.environments.base import Environment
from verl.environments.registry import register


@register("single-turn-math-v0")
class SingleTurnMathEnvironment(Environment):
    async def rollout(self, prompt, server_address, rollout_config, sampling_params):
        client = AsyncOpenAI(base_url=server_address)
        response = await client.chat.completions.create(messages=prompt, **sampling_params).choices[0].message
        gold = parse(rollout_config["ground_truth"])
        answer = parse(response.content)
        reward = 1 if verify(gold, answer) else 0
        return {"messages": prompt + [response], "tools": [], "reward": reward}
