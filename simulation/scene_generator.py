"""
합성 용접 씬 생성기
지원 형상: 철판(flat plate), 파이프(pipe)
지원 이음 유형: 맞대기(butt), T-이음(T-joint), 파이프-판(pipe-on-plate), 파이프-파이프(pipe-butt)
"""

import open3d as o3d
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
from enum import Enum


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────

class JointType(Enum):
    BUTT = "butt"               # 맞대기 이음
    T_JOINT = "t_joint"         # T-이음
    PIPE_ON_PLATE = "pipe_on_plate"   # 파이프-판
    PIPE_BUTT = "pipe_butt"     # 파이프 맞대기


@dataclass
class SceneConfig:
    joint_type: JointType = JointType.BUTT
    num_points: int = 50_000        # 씬 전체 포인트 수
    depth_noise_std: float = 0.001  # 깊이 카메라 노이즈 (m)
    random_seed: int = 42

    # 철판 크기 (m)
    plate_size: Tuple[float, float, float] = (0.3, 0.2, 0.005)

    # 파이프 치수 (m)
    pipe_radius: float = 0.03
    pipe_length: float = 0.25
    pipe_thickness: float = 0.003   # 벽 두께

    # 용접 갭 (m)
    weld_gap: float = 0.002


# ─────────────────────────────────────────────
# 기본 형상 빌더
# ─────────────────────────────────────────────

def _make_plate_mesh(size: Tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
    """size = (x_width, y_depth, z_thickness) — 판은 XY 평면에 배치, Z가 두께"""
    w, d, h = size
    # create_box: width→x, height→y, depth→z
    mesh = o3d.geometry.TriangleMesh.create_box(width=w, height=d, depth=h)
    mesh.translate(-np.array([w / 2, d / 2, h / 2]))
    return mesh


def _make_pipe_mesh(radius: float, length: float, thickness: float,
                    resolution: int = 40) -> o3d.geometry.TriangleMesh:
    """Z축 방향 파이프, 원점 중심 (z: -length/2 ~ +length/2)"""
    outer = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=resolution, split=4)
    # create_cylinder 은 이미 Z 중심 정렬 → 추가 이동 불필요
    return outer


def _sample_and_noise(mesh: o3d.geometry.TriangleMesh, n: int,
                      noise_std: float, rng: np.random.Generator
                      ) -> o3d.geometry.PointCloud:
    pcd = mesh.sample_points_uniformly(number_of_points=n)
    pts = np.asarray(pcd.points)
    pts += rng.normal(0, noise_std, pts.shape)
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


# ─────────────────────────────────────────────
# 용접선(seam) 포인트 생성
# ─────────────────────────────────────────────

def _line_seam(start: np.ndarray, end: np.ndarray, n: int = 200) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    return start + t * (end - start)


def _circle_seam(center: np.ndarray, radius: float,
                 normal_axis: str = 'z', n: int = 300) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n)
    if normal_axis == 'z':
        pts = np.stack([radius * np.cos(theta), radius * np.sin(theta),
                        np.zeros(n)], axis=1)
    elif normal_axis == 'y':
        pts = np.stack([radius * np.cos(theta), np.zeros(n),
                        radius * np.sin(theta)], axis=1)
    return pts + center


# ─────────────────────────────────────────────
# 씬 빌더
# ─────────────────────────────────────────────

def build_butt_joint(cfg: SceneConfig, rng: np.random.Generator):
    """두 철판이 맞대기 이음"""
    w, d, h = cfg.plate_size
    half_gap = cfg.weld_gap / 2

    plate_l = _make_plate_mesh(cfg.plate_size)
    plate_l.translate([-(w / 2 + half_gap), 0, 0])

    plate_r = _make_plate_mesh(cfg.plate_size)
    plate_r.translate([half_gap, 0, 0])

    n_each = cfg.num_points // 2
    pcd_l = _sample_and_noise(plate_l, n_each, cfg.depth_noise_std, rng)
    pcd_r = _sample_and_noise(plate_r, n_each, cfg.depth_noise_std, rng)

    _color(pcd_l, [0.6, 0.6, 0.65])
    _color(pcd_r, [0.6, 0.6, 0.65])

    # 용접선: Z 상단면의 갭 중앙 직선
    seam_pts = _line_seam(
        np.array([0, -d / 2, h / 2]),
        np.array([0,  d / 2, h / 2])
    )

    return [pcd_l, pcd_r], seam_pts


def build_t_joint(cfg: SceneConfig, rng: np.random.Generator):
    """수직 철판이 수평 철판 위에 T자로 이음"""
    w, d, h = cfg.plate_size

    base = _make_plate_mesh(cfg.plate_size)
    # base는 XY 평면에 배치

    vert = _make_plate_mesh((h, d, w * 0.6))   # 세로판: 얇고 키가 큰 형태
    vert_h = w * 0.6
    vert.translate([0, 0, h / 2 + vert_h / 2])

    n_each = cfg.num_points // 2
    pcd_base = _sample_and_noise(base, n_each, cfg.depth_noise_std, rng)
    pcd_vert = _sample_and_noise(vert, n_each, cfg.depth_noise_std, rng)

    _color(pcd_base, [0.55, 0.55, 0.60])
    _color(pcd_vert, [0.50, 0.50, 0.55])

    seam_l = _line_seam(
        np.array([-h / 2, -d / 2, h / 2]),
        np.array([-h / 2,  d / 2, h / 2])
    )
    seam_r = _line_seam(
        np.array([h / 2, -d / 2, h / 2]),
        np.array([h / 2,  d / 2, h / 2])
    )
    seam_pts = np.vstack([seam_l, seam_r])

    return [pcd_base, pcd_vert], seam_pts


def build_pipe_on_plate(cfg: SceneConfig, rng: np.random.Generator):
    """파이프가 철판 위에 Z축으로 수직 세워짐"""
    plate = _make_plate_mesh(cfg.plate_size)
    _, _, ph = cfg.plate_size  # 판 두께

    pipe = _make_pipe_mesh(cfg.pipe_radius, cfg.pipe_length, cfg.pipe_thickness)
    # 파이프 중심을 판 상단 위로 이동 (파이프 하단이 z=ph/2 에 맞닿음)
    pipe.translate([0, 0, ph / 2 + cfg.pipe_length / 2])

    n_each = cfg.num_points // 2
    pcd_plate = _sample_and_noise(plate, n_each, cfg.depth_noise_std, rng)
    pcd_pipe  = _sample_and_noise(pipe,  n_each, cfg.depth_noise_std, rng)

    _color(pcd_plate, [0.6, 0.6, 0.65])
    _color(pcd_pipe,  [0.5, 0.55, 0.6])

    # 용접선: 파이프-판 접촉 원
    seam_pts = _circle_seam(
        center=np.array([0, 0, ph / 2]),
        radius=cfg.pipe_radius,
        normal_axis='z'
    )

    return [pcd_plate, pcd_pipe], seam_pts


def build_pipe_butt(cfg: SceneConfig, rng: np.random.Generator):
    """두 파이프가 Z축 방향으로 맞대기 이음"""
    half_gap = cfg.weld_gap / 2
    half_len = cfg.pipe_length / 2

    pipe_a = _make_pipe_mesh(cfg.pipe_radius, cfg.pipe_length, cfg.pipe_thickness)
    pipe_a.translate([0, 0, -(half_len + half_gap)])   # z: -(length+gap/2) ~ -gap/2

    pipe_b = _make_pipe_mesh(cfg.pipe_radius, cfg.pipe_length, cfg.pipe_thickness)
    pipe_b.translate([0, 0,  (half_len + half_gap)])   # z: +gap/2 ~ +(length+gap/2)

    n_each = cfg.num_points // 2
    pcd_a = _sample_and_noise(pipe_a, n_each, cfg.depth_noise_std, rng)
    pcd_b = _sample_and_noise(pipe_b, n_each, cfg.depth_noise_std, rng)

    _color(pcd_a, [0.55, 0.55, 0.60])
    _color(pcd_b, [0.55, 0.55, 0.60])

    seam_pts = _circle_seam(
        center=np.array([0, 0, 0]),
        radius=cfg.pipe_radius,
        normal_axis='z'
    )

    return [pcd_a, pcd_b], seam_pts


def _color(pcd: o3d.geometry.PointCloud, rgb: list):
    pts = np.asarray(pcd.points)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(rgb, (len(pts), 1))
    )


# ─────────────────────────────────────────────
# 씬 생성 메인 함수
# ─────────────────────────────────────────────

_BUILDERS = {
    JointType.BUTT:         build_butt_joint,
    JointType.T_JOINT:      build_t_joint,
    JointType.PIPE_ON_PLATE: build_pipe_on_plate,
    JointType.PIPE_BUTT:    build_pipe_butt,
}


def generate_scene(cfg: Optional[SceneConfig] = None
                   ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Returns:
        scene_pcd : 합성 포인트클라우드 (전체 씬)
        seam_pts  : 용접선 포인트 배열 (N, 3)
    """
    if cfg is None:
        cfg = SceneConfig()

    rng = np.random.default_rng(cfg.random_seed)
    builder = _BUILDERS[cfg.joint_type]
    pcds, seam_pts = builder(cfg, rng)

    scene_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        scene_pcd += pcd

    return scene_pcd, seam_pts


# ─────────────────────────────────────────────
# 시각화 / 저장 헬퍼
# ─────────────────────────────────────────────

def visualize(scene_pcd: o3d.geometry.PointCloud,
              seam_pts: np.ndarray,
              show_seam: bool = True):
    geometries = [scene_pcd]

    if show_seam:
        seam_pcd = o3d.geometry.PointCloud()
        seam_pcd.points = o3d.utility.Vector3dVector(seam_pts)
        seam_pcd.paint_uniform_color([1, 0.2, 0])   # 빨간색
        geometries.append(seam_pcd)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Welding Scene",
        width=1280, height=720,
        point_show_normal=False,
    )


def save(scene_pcd: o3d.geometry.PointCloud,
         seam_pts: np.ndarray,
         out_dir: str = "output",
         prefix: str = "scene"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    pcd_path  = os.path.join(out_dir, f"{prefix}.pcd")
    seam_path = os.path.join(out_dir, f"{prefix}_seam.npy")

    o3d.io.write_point_cloud(pcd_path, scene_pcd)
    np.save(seam_path, seam_pts)
    print(f"저장 완료: {pcd_path}, {seam_path}")


# ─────────────────────────────────────────────
# CLI 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="용접 합성 씬 생성기")
    parser.add_argument("--joint", choices=[j.value for j in JointType],
                        default=JointType.BUTT.value)
    parser.add_argument("--points", type=int, default=50_000)
    parser.add_argument("--noise", type=float, default=0.001)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = SceneConfig(
        joint_type=JointType(args.joint),
        num_points=args.points,
        depth_noise_std=args.noise,
        random_seed=args.seed,
    )

    print(f"[생성] joint={cfg.joint_type.value}, points={cfg.num_points}")
    scene, seam = generate_scene(cfg)
    print(f"  씬 포인트: {len(scene.points):,}  용접선 포인트: {len(seam)}")

    if args.save:
        save(scene, seam, out_dir="output", prefix=cfg.joint_type.value)

    if not args.no_vis:
        visualize(scene, seam)
