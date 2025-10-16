# inspect_base_agent.py
"""
Inspect BaseAgent class signatures and try a few safe invocation tests.
Run this from project root: python inspect_base_agent.py
"""
import sys, os, inspect, traceback
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

def safe_import(mod, attr=None):
    try:
        m = __import__(mod, fromlist=[attr] if attr else [])
        return getattr(m, attr) if attr else m
    except Exception as e:
        print(f"[import-error] {mod} -> {e}")
        return None

candidates = [
    ("src.core.agent", "BaseAgent"),
    ("src.core.agent_langgraph", "BaseAgent"),
    ("src.core.agent_langgraph", "MongoDBLangGraphAgent"),
]

BaseAgent = None
for mod, name in candidates:
    cls = safe_import(mod, name)
    if cls:
        print(f"[found] {name} from {mod}")
        BaseAgent = cls
        break

if BaseAgent is None:
    print("No BaseAgent found in candidates. Exit.")
    sys.exit(1)

print("\n--- BaseAgent class info ---")
print("repr:", repr(BaseAgent))
try:
    print("module:", BaseAgent.__module__)
    print("qualname:", BaseAgent.__qualname__)
except Exception:
    pass

print("\n--- __init__ signature ---")
try:
    print(inspect.signature(BaseAgent.__init__))
except Exception as e:
    print("Could not get __init__ sig:", e)

print("\n--- Methods and signatures ---")
for name, member in inspect.getmembers(BaseAgent, predicate=inspect.isfunction):
    try:
        print(f"{name} :: {inspect.signature(member)}")
    except Exception:
        print(f"{name} :: <signature unavailable>")

print("\n--- Try minimal instantiation attempts ---")
# build a best-effort dict of args from environment
cfg = {
    "name": "mcp_inspect_agent",
    "model_provider": "openai",
    "model_name": os.getenv("OPENAI_MODEL", "gpt-4o"),
    "tools": [],
}
uri = os.getenv("MONGODB_URI", "")
db_name = os.getenv("MONGODB_DB_NAME", "agent_memory_db")


def try_instantiation():
    tries = []
    # 1: try config object
    try:
        cfg_obj = type("Cfg", (), cfg)()
        print("Trying: BaseAgent(config_obj, None)")
        a = BaseAgent(cfg_obj, None)
        print("Success: BaseAgent(cfg_obj, None)")
        return a
    except Exception as e:
        print("Failed:", repr(e))
        tries.append(("cfg_obj", e))
    # 2: try kwargs mapping (common)
    try:
        print("Trying: BaseAgent(config=cfg, memory_manager=None)")
        a = BaseAgent(config=cfg, memory_manager=None)
        print("Success: BaseAgent(config=cfg, memory_manager=None)")
        return a
    except Exception as e:
        print("Failed:", repr(e))
        tries.append(("cfg_kw", e))
    # 3: try provide mongodb_uri style
    try:
        print("Trying: BaseAgent(mongodb_uri=uri, agent_name=cfg['name'])")
        a = BaseAgent(mongodb_uri=uri, agent_name=cfg["name"], model_provider=cfg["model_provider"], model_name=cfg["model_name"])
        print("Success: BaseAgent(mongodb_uri=..., agent_name=...)")
        return a
    except Exception as e:
        print("Failed:", repr(e))
        tries.append(("uri_kw", e))
    # 4: try positional fallback
    try:
        print("Trying: BaseAgent() - no args")
        a = BaseAgent()
        print("Success: BaseAgent()")
        return a
    except Exception as e:
        print("Failed:", repr(e))
        tries.append(("no_args", e))
    print("All instantiation tries failed. Exceptions:")
    for n, ex in tries:
        print(" -", n, ":", type(ex), ex)
    return None

agent = try_instantiation()

if agent is None:
    print("\nNo usable instance created. Please paste the output above back to the assistant for further guidance.")
    sys.exit(0)

print("\n--- Instance created. Methods on the instance ---")
for name, member in inspect.getmembers(agent, predicate=inspect.ismethod):
    try:
        print(f"{name} :: {inspect.signature(member)}")
    except Exception:
        print(f"{name} :: <signature unavailable>")

print("\n--- Try calling common invocation methods and show full tracebacks ---")
tests = [
    ("ainvoke", {"message": "health check", "user_id": "mcp_test", "session_id": "s1"}),
    ("aexecute", {"message": "health check", "thread_id": "s1"}),
    ("invoke", {"message": "health check", "user_id": "mcp_test", "session_id": "s1"}),
    ("execute", {"message": "health check", "thread_id": "s1"}),
]
for meth_name, kw in tests:
    meth = getattr(agent, meth_name, None)
    if meth is None:
        print(f"{meth_name} not present")
        continue
    print(f"\nCalling {meth_name} with kwargs {kw}...")
    try:
        res = meth(**kw)
        if inspect.isawaitable(res):
            import asyncio
            res = asyncio.get_event_loop().run_until_complete(res)
        print("Result:", res)
    except Exception:
        print("Traceback:")
        traceback.print_exc()
