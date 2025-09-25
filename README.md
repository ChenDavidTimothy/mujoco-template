# MuJoCo Template

Reusable building blocks for MuJoCo-based control projects. The package keeps MuJoCo's native `mj.MjModel`/`mj.MjData` front-and-center while bundling the repetitive plumbing needed to spin up controllers, enforce actuator compatibility, log trajectories, and extract observations.

## Features
- Thin `Env` wrapper that manages resets, optional controllers, reward/done/info hooks, and exposes native model/data handles.
- Fail-fast compatibility checks that keep actuator groups, servo limits, and integrated-velocity policies consistent across controllers and models.
- Declarative observation layer that validates named entities and returns either structured dicts or flattened arrays.
- On-demand linearization and Jacobian utilities with native MuJoCo fallbacks for fast prototyping of optimal control pipelines.
- Logging, rollout, and runtime helpers that eliminate boilerplate for headless runs, viewer sessions, and CSV trajectory capture.

## Installation
```bash
pip install mujoco-template
```

The package targets Python 3.10+ and MuJoCo 3.1+. Make sure the MuJoCo runtime libraries are available on your system (see the official MuJoCo installation guide) before installing.

For development work:
```bash
pip install -e .[test]
```

## Quick Start
```python
from mujoco_template import ModelHandle, Env, ObservationSpec, ZeroController

handle = ModelHandle.from_xml_path("examples/pendulum/pendulum.xml")
obs_spec = ObservationSpec(include_sensordata=False)
controller = ZeroController()

env = Env(handle, controller=controller, obs_spec=obs_spec)
obs0 = env.reset()
result = env.step()
print(f"qpos after one step: {result.obs['qpos']}")
```

Prefer a shortcut rollout? Use `quick_rollout`:
```python
from mujoco_template import ObservationSpec, ZeroController, quick_rollout

trajectory = quick_rollout(
    "examples/cartpole/cartpole.xml",
    steps=200,
    controller=ZeroController(),
    obs_spec=ObservationSpec(include_sensordata=False),
)
print(f"Collected {len(trajectory)} observations")
```

Run the built-in smoke-test CLI:
```bash
python -m mujoco_template path/to/model.xml --steps 300 --zero
```

## Controllers and Compatibility
Controllers implement the `Controller` protocol (`prepare`, `__call__`, and a `ControllerCapabilities` dataclass). Capabilities advertise control space, actuator-group requirements, and whether linearizations or Jacobians are requested. `Env` consults `check_controller_compat` to ensure the model, enabled actuators, and controller expectations line up. Strictness is configurable via `strict_servo_limits` and `strict_intvelocity_actrange` flags when constructing `Env` or calling `quick_rollout`.

## Observations
`ObservationSpec` enumerates exactly which signals to extract (qpos, qvel, ctrl, sites, bodies, geoms, subtree COM, etc.). `ObservationExtractor` validates every named site/body and raises early if the model lacks the requested data. Set `as_dict=False` to receive a flattened `numpy` array instead of a dict.

## Linearization and Jacobians
When a controller declares `needs_linearization` or `needs_jacobians`, `Env.step` precomputes discrete-time `(A, B)` matrices and requested Jacobians immediately after the controller writes controls and before advancing the simulation. Results are surfaced once per step through the `StepResult.info` dict. You can also call `env.linearize()` directly to reuse the native-or-fallback finite difference logic.

## Logging and Runtime Helpers
`TrajectoryLogger` outputs CSV trajectories with custom formatting logic. `StateControlRecorder` provides a ready-to-use `StepHook` that records time, qpos, qvel, controls, and optional derived probes on every step. The runtime module also includes `iterate_passive`, `run_passive_headless`, and `run_passive_viewer` to drive simulations in headless or interactive viewer modes.

## Examples
Self-contained MJCF examples live under `examples/`:
- `examples/pendulum` - passive pendulum demos.
- `examples/cartpole` - cart-pole models and scripts.

Each example directory contains the MJCF file plus convenience scripts that showcase how to assemble controllers, observation specs, and runtime utilities.

## Testing
Run the test suite with:
```bash
pytest
```

## Contributing
Issues and pull requests are welcome. Please keep the public API thin, add targeted tests for new features, and run `pytest` before submitting changes. The project uses `mypy --strict`; follow the existing typing discipline when contributing.
