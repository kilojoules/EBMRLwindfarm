import safety_gymnasium
env = safety_gymnasium.make("SafetyPointGoal1-v0")
env.reset(seed=1)

def dump(label):
    print(f"--- {label}")
    e = env
    i = 0
    while e is not None:
        es = getattr(e, '_elapsed_steps', 'N/A')
        mx = getattr(e, '_max_episode_steps', 'N/A')
        print(f"  L{i} {type(e).__name__}  _elapsed_steps={es}  _max_episode_steps={mx}")
        e = getattr(e, 'env', None)
        i += 1

dump("after reset")
for _ in range(3):
    env.step(env.action_space.sample())
dump("after 3 steps")
