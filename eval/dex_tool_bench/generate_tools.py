"""Generate multiple primitive tools for the dex_tool_bench.

This script defines various tool configurations and generates both URDF and OBJ files
for each tool using the functions from create_obj_urdf.py.
"""

from pathlib import Path
from typing import List

from create_obj_urdf import (
    Cuboid,
    Cylinder,
    ToolConfig,
    create_tool,
)
import shutil


# Base output directory for all generated tools
BASE_OUTPUT_DIR = Path("/share/portal/kk837/sapg/assets/urdf/dex_tool_bench")


# =============================================================================
# Hammer Configurations
# =============================================================================

HAMMER_CONFIGS: List[ToolConfig] = [
    # Cuboidal hammer: cuboid handle + cuboid head
    ToolConfig(
        name="cuboidal_hammer",
        handle=Cuboid(length=0.25, width=0.03, height=0.02),
        head=Cuboid(length=0.02, width=0.11, height=0.02),
        tool_type="hammer",
    ),
    # Cuboidal mallet: cuboid handle + cuboid head (wider/thicker head)
    ToolConfig(
        name="cuboidal_mallet",
        handle=Cuboid(length=0.24, width=0.03, height=0.02),
        head=Cuboid(length=0.05, width=0.08, height=0.045),
        tool_type="hammer",
    ),
]


# =============================================================================
# Screwdriver Configurations
# =============================================================================

SCREWDRIVER_CONFIGS: List[ToolConfig] = [
    # Cuboidal screwdriver: cuboid handle (grip) + cuboid head (flat shaft)
    ToolConfig(
        name="cuboidal_screwdriver",
        handle=Cuboid(length=0.15, width=0.025, height=0.025),  # Compact grip
        head=Cuboid(length=0.12, width=0.006, height=0.006),    # Thin flat shaft
        tool_type="screwdriver",
    ),
    # Cylindrical screwdriver: cylinder handle (grip) + cuboid head (flat shaft)
    ToolConfig(
        name="cylindrical_screwdriver",
        handle=Cylinder(length=0.18, radius=0.015),  # Round grip
        head=Cuboid(length=0.10, width=0.006, height=0.006),  # Thin flat shaft
        tool_type="screwdriver",
    ),
]


# =============================================================================
# Eraser Configurations
# =============================================================================

ERASER_CONFIGS: List[ToolConfig] = [
    # Small whiteboard eraser: thickness along Y (width)
    ToolConfig(
        name="small_eraser",
        handle=Cuboid(length=0.10, width=0.025, height=0.05),  # 10cm long, 2.5cm thick, 5cm tall
        head=Cuboid(length=0.001, width=0.001, height=0.001),  # Near-zero head
        tool_type="eraser",
    ),
    # Large whiteboard eraser: thickness along Y (width)
    ToolConfig(
        name="large_eraser",
        handle=Cuboid(length=0.12, width=0.03, height=0.06),   # 12cm long, 3cm thick, 6cm tall
        head=Cuboid(length=0.001, width=0.001, height=0.001),  # Near-zero head
        tool_type="eraser",
    ),
]


# =============================================================================
# Spatula Configurations
# =============================================================================

SPATULA_CONFIGS: List[ToolConfig] = [
    # Small spatula: thin handle + narrow flat blade
    ToolConfig(
        name="small_spatula",
        handle=Cuboid(length=0.15, width=0.02, height=0.015),  # Thin grip handle
        head=Cuboid(length=0.08, width=0.06, height=0.003),    # Flat narrow blade
        tool_type="spatula",
    ),
    # Large spatula: cylinder handle + wide flat blade
    ToolConfig(
        name="large_spatula",
        handle=Cylinder(length=0.18, radius=0.012),  # Round grip handle
        head=Cuboid(length=0.10, width=0.10, height=0.004),    # Flat wide blade
        tool_type="spatula",
    ),
]


# =============================================================================
# Marker Configurations
# =============================================================================

MARKER_CONFIGS: List[ToolConfig] = [
    # Thin marker: slim cylinder body + small tip
    ToolConfig(
        name="thin_marker",
        handle=Cylinder(length=0.12, radius=0.006),  # Slim marker body
        head=Cuboid(length=0.015, width=0.004, height=0.004),  # Small tip
        tool_type="marker",
    ),
    # Thick marker: chunky cylinder body + larger tip
    ToolConfig(
        name="thick_marker",
        handle=Cylinder(length=0.14, radius=0.012),  # Chunky marker body
        head=Cuboid(length=0.02, width=0.008, height=0.008),  # Larger tip
        tool_type="marker",
    ),
]


# =============================================================================
# Knife Configurations
# =============================================================================

KNIFE_CONFIGS: List[ToolConfig] = [
    # Kitchen knife: cuboid handle + thin flat blade
    ToolConfig(
        name="kitchen_knife",
        handle=Cuboid(length=0.12, width=0.025, height=0.02),  # Grip handle
        head=Cuboid(length=0.18, width=0.04, height=0.002),    # Thin flat blade
        tool_type="knife",
    ),
    # Paring knife: cylinder handle + shorter blade
    ToolConfig(
        name="paring_knife",
        handle=Cylinder(length=0.10, radius=0.012),  # Round grip handle
        head=Cuboid(length=0.10, width=0.025, height=0.002),   # Shorter thin blade
        tool_type="knife",
    ),
]


# =============================================================================
# All Tool Configurations
# =============================================================================

TOOL_CONFIGS: List[ToolConfig] = [
    *HAMMER_CONFIGS,
    *SCREWDRIVER_CONFIGS,
    *ERASER_CONFIGS,
    *SPATULA_CONFIGS,
    *MARKER_CONFIGS,
    *KNIFE_CONFIGS,
]


# =============================================================================
# Generation
# =============================================================================

def generate_all_tools() -> None:
    """Generate all tool configurations."""
    print("=" * 60)
    print("Generating tools")
    print("=" * 60)
    
    for config in TOOL_CONFIGS:
        print(f"\n--- {config.name} ---")
        output_dir = BASE_OUTPUT_DIR / config.tool_type / config.name
        # if output_dir already exists, erase it
        if output_dir.exists():
            shutil.rmtree(output_dir)
        create_tool(output_dir, config)
    
    print("\n" + "=" * 60)
    print(f"Generated {len(TOOL_CONFIGS)} tools")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_tools()
