# Optimality Review

## Verdict
Optimal.

## Rationale
- Observation extraction now defaults to zero-copy views, keeping high-rate controllers on the metal while leaving an opt-in escape hatch for detached buffers.
- Actuator groups remain under researcher control; `Env` no longer mutates availability automatically, instead surfacing advisory warnings so the model state stays the single source of truth.
- Compatibility checks report servo metadata gaps, activation/force limit quirks, and group mismatches as warnings without blocking execution, so advanced workflows retain MuJoCo's native flexibility while still seeing actionable telemetry.
