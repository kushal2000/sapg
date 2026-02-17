"""Visualize all tools in the dex_tool_bench_training folder using viser.

This script finds all URDF and OBJ files in the dex_tool_bench_training assets folder
and displays them organized by tool type with section titles.

Handle is colored with wooden color, head with gray metallic color.
"""

import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from isaacgymenvs.utils.utils import get_repo_root_dir

import numpy as np
import trimesh
import viser


# Base directory for training tools assets
ASSETS_DIR = get_repo_root_dir() / "assets/urdf/dex_tool_bench_training"

# Colors (RGB 0-255)
WOOD_COLOR = (139, 90, 43)  # Dark walnut brown
METAL_COLOR = (105, 105, 105)  # Dark steel gray
KEYPOINT_COLOR = (255, 50, 50)  # Bright red for keypoints

# Keypoint configuration
KEYPOINT_RADIUS = 0.006  # Radius of keypoint spheres
KEYPOINT_SCALE = 1.0  # Scale factor for keypoint positions (1.0 = at handle corners)
KEYPOINT_OFFSETS = [
    [1, 1, 1],
    [1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1],
]  # Normalized keypoint positions (corners of bounding box)


def parse_urdf_geometry(urdf_path: Path) -> Tuple[Optional[dict], Optional[dict], Optional[float]]:
    """Parse URDF file to extract handle and head geometry parameters.
    
    Returns:
        (handle_params, head_params, head_offset_x)
        handle_params: dict with 'type' ('box' or 'cylinder') and dimensions
        head_params: dict with 'type' ('box' or 'cylinder') and dimensions
        head_offset_x: x offset of head from origin
    """
    with open(urdf_path, 'r') as f:
        content = f.read()
    
    # Find all visual elements
    visual_pattern = r'<visual>\s*<origin xyz="([^"]+)"[^>]*/>.*?<geometry>\s*(.*?)\s*</geometry>'
    visuals = re.findall(visual_pattern, content, re.DOTALL)
    
    if len(visuals) < 2:
        return None, None, None
    
    handle_params = None
    head_params = None
    head_offset = None
    
    for origin_xyz, geometry in visuals:
        xyz = [float(x) for x in origin_xyz.split()]
        x_offset = xyz[0]
        
        # Parse geometry
        box_match = re.search(r'<box size="([^"]+)"/>', geometry)
        cylinder_match = re.search(r'<cylinder length="([^"]+)" radius="([^"]+)"/>', geometry)
        
        params = None
        if box_match:
            sizes = [float(s) for s in box_match.group(1).split()]
            params = {'type': 'box', 'length': sizes[0], 'width': sizes[1], 'height': sizes[2]}
        elif cylinder_match:
            params = {'type': 'cylinder', 'length': float(cylinder_match.group(1)), 'radius': float(cylinder_match.group(2))}
        
        if params:
            if abs(x_offset) < 0.001:  # Handle is at origin
                handle_params = params
            else:  # Head is offset
                head_params = params
                head_offset = x_offset
    
    return handle_params, head_params, head_offset


def create_mesh_from_params(params: dict, is_head: bool = False) -> Optional[trimesh.Trimesh]:
    """Create a trimesh from geometry parameters."""
    if params is None:
        return None
    
    if params['type'] == 'box':
        mesh = trimesh.creation.box(extents=(params['length'], params['width'], params['height']))
    elif params['type'] == 'cylinder':
        mesh = trimesh.creation.cylinder(radius=params['radius'], height=params['length'])
        if is_head:
            # Rotate 90 degrees around X axis to align with Y axis
            rotation = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
        else:
            # Rotate 90 degrees around Y axis to align with X axis
            rotation = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0])
        mesh.apply_transform(rotation)
    else:
        return None
    
    return mesh


def compute_keypoints(handle_params: dict) -> List[Tuple[float, float, float]]:
    """Compute keypoint positions based on handle geometry.
    
    Keypoints are placed at corners of the handle bounding box.
    For cylinder handles, we use the bounding box of the cylinder.
    
    Returns:
        List of (x, y, z) keypoint positions relative to handle center.
    """
    if handle_params is None:
        return []
    
    if handle_params['type'] == 'box':
        # Half-extents for cuboid
        hx = handle_params['length'] / 2
        hy = handle_params['width'] / 2
        hz = handle_params['height'] / 2
    elif handle_params['type'] == 'cylinder':
        # Cylinder is aligned along X axis after rotation
        hx = handle_params['length'] / 2
        hy = handle_params['radius']
        hz = handle_params['radius']
    else:
        return []
    
    # Generate keypoints at corners based on normalized offsets (with 1.5x scale)
    keypoints = []
    for offset in KEYPOINT_OFFSETS:
        kp = (
            offset[0] * hx * KEYPOINT_SCALE,
            offset[1] * hy * KEYPOINT_SCALE,
            offset[2] * hz * KEYPOINT_SCALE,
        )
        keypoints.append(kp)
    
    return keypoints


def find_all_tools_by_type(base_dir: Path) -> Dict[str, List[Path]]:
    """Find all tools grouped by tool type (parent folder).
    
    Returns:
        Dict mapping tool_type -> list of urdf_paths.
    """
    tools_by_type = defaultdict(list)
    urdf_files = sorted(base_dir.rglob("*.urdf"))
    
    for urdf_path in urdf_files:
        # Get tool type from folder structure: base_dir / tool_type / tool_name / file.urdf
        relative_path = urdf_path.relative_to(base_dir)
        if len(relative_path.parts) >= 2:
            tool_type = relative_path.parts[0]
        else:
            tool_type = "other"
        
        tools_by_type[tool_type].append(urdf_path)
    
    return dict(tools_by_type)


def main() -> None:
    """Visualize all tools in the dex_tool_bench_training folder."""
    
    # Find all tools grouped by type
    tools_by_type = find_all_tools_by_type(ASSETS_DIR)
    
    if not tools_by_type:
        print(f"No tools found in {ASSETS_DIR}")
        return
    
    total_tools = sum(len(tools) for tools in tools_by_type.values())
    print(f"Found {total_tools} tools in {len(tools_by_type)} categories:")
    for tool_type, tools in tools_by_type.items():
        print(f"  [{tool_type.upper()}]: {len(tools)} tools")
    
    # Start viser server
    server = viser.ViserServer(port=8080)
    print(f"\nViser server running at http://localhost:8080")
    
    # Layout parameters - adjusted for more tools
    spacing_y = 0.25  # Space between tools along Y axis (tighter for more tools)
    section_spacing_x = 0.65  # Space between sections along X axis
    title_height = 0.15  # Height of title above tools
    x_offset = -0.5  # Shift all objects back in X direction
    
    current_x = x_offset
    
    # Load tools by section (sections along X, tools along Y)
    for tool_type, tools in sorted(tools_by_type.items()):
        print(f"\nLoading section: {tool_type} ({len(tools)} tools)")
        
        # Add section title at the front of the section
        server.scene.add_label(
            f"/section_{tool_type}/title",
            text=f"{tool_type.upper().replace('_', ' ')} ({len(tools)})",
            position=(current_x, -0.15, title_height),
        )
        
        # Load each tool in this section (along Y axis)
        for i, urdf_path in enumerate(tools):
            y_pos = i * spacing_y
            
            tool_name = urdf_path.stem
            print(f"  [{i+1}/{len(tools)}] Loading: {tool_name}")
            
            # Parse URDF to get handle and head geometry
            handle_params, head_params, head_offset = parse_urdf_geometry(urdf_path)
            
            # Create frame for this tool
            server.scene.add_frame(
                f"/section_{tool_type}/{tool_name}",
                position=(current_x, y_pos, 0.0),
                axes_length=0.03,
                axes_radius=0.001,
            )
            
            # Create and add handle mesh (wooden color)
            if handle_params:
                handle_mesh = create_mesh_from_params(handle_params, is_head=False)
                if handle_mesh:
                    handle_mesh.visual.face_colors = [*WOOD_COLOR, 255]
                    server.scene.add_mesh_trimesh(
                        name=f"/section_{tool_type}/{tool_name}/handle",
                        mesh=handle_mesh,
                    )
            
            # Create and add head mesh (metallic gray color)
            if head_params and head_offset:
                head_mesh = create_mesh_from_params(head_params, is_head=True)
                if head_mesh:
                    head_mesh.apply_translation([head_offset, 0, 0])
                    head_mesh.visual.face_colors = [*METAL_COLOR, 255]
                    server.scene.add_mesh_trimesh(
                        name=f"/section_{tool_type}/{tool_name}/head",
                        mesh=head_mesh,
                    )
            
            # Add keypoints as small spheres around the handle
            if handle_params:
                keypoints = compute_keypoints(handle_params)
                for kp_idx, kp_pos in enumerate(keypoints):
                    server.scene.add_icosphere(
                        name=f"/section_{tool_type}/{tool_name}/keypoint_{kp_idx}",
                        radius=KEYPOINT_RADIUS,
                        color=KEYPOINT_COLOR,
                        position=kp_pos,
                    )
        
        # Move to next section along X axis
        current_x += section_spacing_x
    
    # Add grid for reference
    server.scene.add_grid(
        "/grid",
        width=8,
        height=10,
        position=(1.0, 2.0, -0.01),
    )
    
    print(f"\nLoaded {total_tools} tools in {len(tools_by_type)} sections")
    print("Press Ctrl+C to exit.")
    
    # Keep the server running
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()

