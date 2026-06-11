"""Probe safety_gymnasium env to find mujoco state accessors."""
import safety_gymnasium, torch
env = safety_gymnasium.make("SafetyPointGoal1-v0")
env.reset(seed=1)

def explore(obj, path="env", depth=0, max_depth=4):
    if depth > max_depth: return
    t = type(obj).__name__
    interesting = [a for a in dir(obj) if not a.startswith('_') and
                   any(k in a.lower() for k in ['sim','data','qpos','qvel','state','model','agent','task','engine'])]
    print(f"{'  '*depth}{path}: {t}  attrs={interesting[:15]}")
    for a in ['unwrapped', 'env', 'task', 'agent', 'sim', 'data', 'engine']:
        if hasattr(obj, a):
            try:
                sub = getattr(obj, a)
                explore(sub, f"{path}.{a}", depth+1, max_depth)
            except Exception:
                pass

explore(env)

# Direct attempts
print("\n=== direct attempts ===")
for path_str in [
    "env.unwrapped.data.qpos",
    "env.unwrapped.data.qvel",
    "env.unwrapped.sim.data.qpos",
    "env.unwrapped.task.data.qpos",
    "env.unwrapped.task.agent.pos",
    "env.unwrapped.task.agent.vel",
    "env.unwrapped.task.engine.data.qpos",
    "env.unwrapped.task.engine.sim.get_state()",
]:
    try:
        v = eval(path_str)
        print(f"  OK  {path_str}  shape={getattr(v,'shape',None) or getattr(v,'__len__',lambda:None)()}")
    except Exception as e:
        print(f"  FAIL {path_str}  ({type(e).__name__}: {str(e)[:80]})")
