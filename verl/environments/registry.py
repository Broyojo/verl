ENVIRONMENT_REGISTRY = {}


def register(name):
    def decorator(cls):
        if name in ENVIRONMENT_REGISTRY and ENVIRONMENT_REGISTRY[name] != cls:
            raise ValueError(f"Environment {name} has already been registered: {ENVIRONMENT_REGISTRY[name]} vs {cls}")
        ENVIRONMENT_REGISTRY[name] = cls
        return cls

    return decorator


def get_reward_manager_cls(name):
    if name not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unknown environment: {name}")
    return ENVIRONMENT_REGISTRY[name]
