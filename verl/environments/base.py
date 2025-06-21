from abc import ABC, abstractmethod


class Environment(ABC):
    @abstractmethod
    async def rollout(self, prompt, server_address, rollout_config, sampling_params):
        pass
