# Type stub for mujoco module
# This file provides type hints for the mujoco module to resolve Pylance errors

from typing import Any
import numpy as np

# Basic types that are commonly used
class MjModel:
    nu: int
    nq: int
    nv: int
    nkey: int
    nsensordata: int
    actuator_group: np.ndarray[tuple[int, ...], np.dtype[np.int32]]
    actuator_ctrlrange: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    actuator_ctrllimited: np.ndarray[tuple[int, ...], np.dtype[np.bool_]]
    actuator_actlimited: np.ndarray[tuple[int, ...], np.dtype[np.bool_]]
    actuator_actrange: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    actuator_forcelimited: np.ndarray[tuple[int, ...], np.dtype[np.bool_]]
    actuator_forcerange: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    opt: Any
    
    @classmethod
    def from_xml_path(cls, xml_path: str) -> "MjModel": ...
    
    @classmethod
    def from_xml_string(cls, xml_text: str) -> "MjModel": ...
    
    @classmethod
    def from_binary_path(cls, mjb_path: str) -> "MjModel": ...

class MjData:
    qpos: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    qvel: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    qacc: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    ctrl: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    act: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    time: float
    sensordata: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    site_xpos: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    xpos: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    xipos: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    geom_xpos: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    subtree_com: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    qfrc_inverse: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    actuator_moment: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    moment_rownnz: np.ndarray[tuple[int, ...], np.dtype[np.int32]]
    moment_rowadr: np.ndarray[tuple[int, ...], np.dtype[np.int32]]
    moment_colind: np.ndarray[tuple[int, ...], np.dtype[np.int32]]
    
    def __init__(self, model: MjModel) -> None: ...

class mjtObj:
    mjOBJ_SITE: int
    mjOBJ_BODY: int
    mjOBJ_GEOM: int
    mjOBJ_KEY: int

# Function stubs
def mj_name2id(model: MjModel, objtype: int, name: str) -> int: ...
def mj_forward(model: MjModel, data: MjData) -> None: ...
def mj_step(model: MjModel, data: MjData) -> None: ...
def mj_resetData(model: MjModel, data: MjData) -> None: ...
def mj_resetDataKeyframe(model: MjModel, data: MjData, key: int) -> None: ...
def mj_subtreeCoM(model: MjModel, data: MjData) -> None: ...
def mj_differentiatePos(model: MjModel, dq: np.ndarray[tuple[int, ...], np.dtype[np.float64]], dt: float, qpos2: np.ndarray[tuple[int, ...], np.dtype[np.float64]], qpos1: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> None: ...
def mj_integratePos(model: MjModel, qpos: np.ndarray[tuple[int, ...], np.dtype[np.float64]], dq: np.ndarray[tuple[int, ...], np.dtype[np.float64]], dt: float) -> None: ...
def mj_inverse(model: MjModel, data: MjData) -> None: ...
def mj_jacSite(model: MjModel, data: MjData, jacp: np.ndarray[tuple[int, ...], np.dtype[np.float64]], jacr: np.ndarray[tuple[int, ...], np.dtype[np.float64]], site_id: int) -> None: ...
def mj_jacBody(model: MjModel, data: MjData, jacp: np.ndarray[tuple[int, ...], np.dtype[np.float64]], jacr: np.ndarray[tuple[int, ...], np.dtype[np.float64]], body_id: int) -> None: ...
def mj_jacBodyCom(model: MjModel, data: MjData, jacp: np.ndarray[tuple[int, ...], np.dtype[np.float64]], body_id: int) -> None: ...
def mj_jacSubtreeCom(model: MjModel, data: MjData, jacp: np.ndarray[tuple[int, ...], np.dtype[np.float64]], body_id: int) -> None: ...
def mj_saveModel(model: MjModel, path: str, buffer: Any | None = None) -> None: ...
def mjd_transitionFD(model: MjModel, data: MjData, eps: float, centered: bool, A: np.ndarray[tuple[int, ...], np.dtype[np.float64]], B: np.ndarray[tuple[int, ...], np.dtype[np.float64]], *args: Any) -> None: ...
def mju_sparse2dense(dense: np.ndarray[tuple[int, ...], np.dtype[np.float64]], sparse: np.ndarray[tuple[int, ...], np.dtype[np.float64]], rownnz: np.ndarray[tuple[int, ...], np.dtype[np.int32]], rowadr: np.ndarray[tuple[int, ...], np.dtype[np.int32]], colind: np.ndarray[tuple[int, ...], np.dtype[np.int32]]) -> None: ...
