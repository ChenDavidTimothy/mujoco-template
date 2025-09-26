# MuJoCo Template

Reusable building blocks for MuJoCo-based control projects. The package keeps MuJoCo's native `mj.MjModel`/`mj.MjData` front-and-center while bundling the repetitive plumbing needed to spin up controllers, enforce actuator compatibility, log trajectories, and extract observations.

## Features
- Thin `Env` wrapper that manages resets, optional controllers, reward/done/info hooks, and exposes native model/data handles.
- Compatibility telemetry that surfaces actuator group mismatches, servo metadata gaps, and integrated-velocity policy limits as warnings without blocking execution.
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
obs_spec = ObservationSpec()
controller = ZeroController()

env = Env(handle, controller=controller, obs_spec=obs_spec)
obs0 = env.reset()
result = env.step()
print(f"qpos after one step: {result.obs['qpos']}")
```


Prefer a shortcut rollout? Use `rollout`:
```python
from mujoco_template import ObservationSpec, ZeroController, rollout

trajectory = rollout(
    "examples/cartpole/cartpole.xml",
    steps=200,
    controller=ZeroController(),
    obs_spec=ObservationSpec(),
)
print(f"Collected {len(trajectory)} observations")
```

Need zero-boilerplate runtime tooling? Configure once and reuse `PassiveRunHarness`:
```python
import mujoco_template as mt

harness = mt.PassiveRunHarness(
    build_env,  # returns an mt.Env
    description="My MuJoCo experiment",
    seed_fn=seed_env,  # optional callable to set initial state
    probes=make_probes,  # optional callable returning DataProbe instances
)

settings = mt.PassiveRunSettings(
    simulation=mt.SimulationSettings(max_steps=2_000),
    logging=mt.LoggingSettings(enabled=True, path="trajectory.csv"),
)

result = harness.run_from_cli(settings)
print(f"Simulation executed {result.steps} steps")
```

Run the built-in smoke-test CLI:
```bash
python -m mujoco_template path/to/model.xml --steps 300 --zero
```

## Controllers and Compatibility
Controllers implement the `Controller` protocol (`prepare`, `__call__`, and a `ControllerCapabilities` dataclass). Capabilities advertise control space, actuator-group requirements, and whether linearizations or Jacobians are requested. `Env` consults `check_controller_compat` to report on the model, enabled actuators, and controller expectations; actuator-group mismatches, servo metadata gaps, and activation/force limit quirks now produce warnings while leaving MuJoCo's native behaviour untouched. Opt into actuator disabling explicitly via `enabled_groups` when you really need it—the environment no longer changes group availability on your behalf.

## Observations
`ObservationSpec` enumerates exactly which signals to extract (qpos, qvel, ctrl, sites, bodies, geoms, subtree COM, etc.). `ObservationExtractor` validates every named site/body and raises early if the model lacks the requested data. Sensor streams and simulation time are opt-in, and the `copy` flag now defaults to `False` so advanced controllers receive zero-copy views of MuJoCo's state. Leave it there for throughput, or flip it to `True` when downstream code keeps observation arrays around between steps (for example, when logging raw observations asynchronously) and therefore needs stable copies. Set `as_dict=False` to receive a flattened `numpy` array instead of a dict.

## Linearization and Jacobians
When a controller declares `needs_linearization` or `needs_jacobians`, `Env.step` precomputes discrete-time `(A, B)` matrices and requested Jacobians immediately after the controller writes controls and before advancing the simulation. Results are surfaced once per step through the `StepResult.info` dict. You can also call `env.linearize()` directly to reuse the native-or-fallback finite difference logic.

## Logging and Runtime Helpers
`TrajectoryLogger` outputs CSV trajectories with custom formatting logic. `StateControlRecorder` provides a ready-to-use `StepHook` that records time, qpos, qvel, controls, and optional derived probes on every step. The runtime module also includes `iterate_passive`, `run_passive_headless`, and `run_passive_viewer` to drive simulations in headless or interactive viewer modes.

### Adaptive Framing Camera for Video Export

Programmatic video exports now support an adaptive framing camera. Populate `VideoSettings.adaptive_camera` with an `AdaptiveCameraSettings` instance to enable distance-, FOV-, or orthographic-based zoom control while keeping the camera orientation fixed. The controller tracks named points of interest (`"body:<name>"`, `"site:<name>"`, `"geom:<name>"`, or COM variants), expands or contracts the shot using configurable hysteresis thresholds, and applies exponential smoothing and optional recentering for stable footage. Supply `recenter_axis` with a single axis (for example `"y"`), an iterable such as `("x", "z")`, or a comma/whitespace separated string like `"x, z"` to recenter along multiple axes. Bounds, safety margins, smoothing constants, and zoom policy are all configured explicitly, and the feature is entirely opt-in—the previous behaviour remains unchanged when the adaptive camera is not supplied.

## Examples
Self-contained MJCF examples live under `examples/`:
- `examples/pendulum` - passive pendulum demos.
- `examples/cartpole` - cart-pole models and scripts.
- `examples/humanoid` - LQR balancing controller for the MuJoCo humanoid.

Each example directory contains the MJCF file plus convenience scripts that showcase how to assemble controllers, observation specs, and runtime utilities. Runtime defaults now live in dedicated config modules (for example `examples/cartpole/cartpole_config.py`). The scripts only accept the activation flags `--viewer`, `--video`, and `--logs`; all other parameters are configured in the companion config file so the source of truth is explicit.

## Testing
Run the test suite with:
```bash
pytest
```

## Contributing
Issues and pull requests are welcome. Please keep the public API thin, add targeted tests for new features, and run `pytest` before submitting changes. The project uses `mypy --strict`; follow the existing typing discipline when contributing.
