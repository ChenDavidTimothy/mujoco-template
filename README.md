# mujoco-template

Reusable building blocks for MuJoCo-based control projects. Version 3 promotes a
single high-level façade—`SimulationSession`—that loads models, wires
controllers, enforces actuator compatibility, and exposes MuJoCo-native
`model`/`data` handles without the repetitive plumbing that previously lived in
every example.

## Features
- `SimulationSession` headless runner that hides `ModelHandle`/`Env` wiring while
  keeping MuJoCo handles directly accessible.
- Callable-to-controller adapter so quick experiments can provide a simple
  function instead of implementing the full protocol.
- Declarative observation presets with fluent builders that cover common signal
  bundles (`qpos`, `qvel`, time, control, named sites/bodies) in one line.
- Compatibility checks, Jacobian/linearization utilities, and actuator-group
  enforcement remain available through the façade for optimal-control workflows.

## Installation
```bash
pip install mujoco-template
```

The package targets Python 3.10+ and MuJoCo 3.1+. Make sure the MuJoCo runtime
libraries are available on your system (see the official MuJoCo installation
guide) before installing.

For development work:
```bash
pip install -e .[test]
```

## Quick Start
```python
from mujoco_template import ObservationSpec, SimulationSession, controller_from_callable

session = SimulationSession.from_xml_path(
    "examples/pendulum/pendulum.xml",
    controller=controller_from_callable(lambda _m, data, _t: data.ctrl.fill(0.0)),
    observation=ObservationSpec.basic().with_time().with_ctrl(),
)

obs0 = session.reset()
result = session.step()
print(f"qpos after one step: {result.obs['qpos']}")

trajectory = session.run(duration_seconds=1.0, sample_stride=10)
print(f"Executed {trajectory.steps} simulation steps")
```

## Controllers and Compatibility
Controllers still implement the strict protocol (`prepare`, `__call__`, and a
`ControllerCapabilities` dataclass). For quick experiments, wrap a callable with
`controller_from_callable`. Capabilities advertise control space,
actuator-group requirements, and whether linearizations or Jacobians are
requested. `SimulationSession` performs compatibility checks during
construction—actuator mismatches emit warnings and remain discoverable through
`session.compat_warnings()`.

## Observations
`ObservationSpec` now includes presets and fluent builders:

```python
spec = (
    ObservationSpec.basic()
    .with_sites("tip")
    .with_bodies("torso", inertial=True)
    .with_time()
)
```

Use tokens for even terser declarations: `ObservationSpec.from_tokens(["qpos",
"qvel", "site:tip"])`. `ObservationExtractor` still validates every named
entity and can return dicts or flattened arrays (`spec.as_array()`).

## Examples
Self-contained MJCF examples live under `examples/` and now import the façade
instead of the legacy environment classes. Each directory contains an MJCF file
plus a Python script that builds a `SimulationSession` and demonstrates the new
API.

## Testing
Run the test suite with:
```bash
pytest
```

## Contributing
Issues and pull requests are welcome. Please keep the public API thin, add
targeted tests for new features, and run `pytest` before submitting changes.
