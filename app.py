# app.py

import streamlit as st
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
import itertools
import heapq
from rasterio.transform import from_bounds

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(layout="wide", page_title="Drone Mission Planner")

# --- Helper Functions (A*, Node Class, etc.) ---
# These are the same functions from your notebook
class Node:
    def __init__(self, parent=None, position=None): self.parent, self.position, self.g, self.h, self.f = parent, position, 0, 0, 0
    def __eq__(self, other): return self.position == other.position
    def __lt__(self, other): return self.f < other.f
    def __hash__(self): return hash(self.position)

def astar(grid, start, end):
    open_list, closed_set = [], set()
    start_node, end_node = Node(None, start), Node(None, end)
    heapq.heappush(open_list, start_node)
    while len(open_list) > 0:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node)
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
            if not (0 <= node_position[0] < len(grid) and 0 <= node_position[1] < len(grid[0])): continue
            if grid[node_position[0]][node_position[1]] < 0: continue
            new_node = Node(current_node, node_position)
            children.append(new_node)
        for child in children:
            if child in closed_set: continue
            child.g = current_node.g + grid[child.position[0]][child.position[1]]
            child.h = abs(child.position[0] - end_node.position[0]) + abs(child.position[1] - end_node.position[1])
            child.f = child.g + child.h
            if any(open_node for open_node in open_list if child == open_node and child.g > open_node.g): continue
            heapq.heappush(open_list, child)
    return None # Path not found

def find_nearest_walkable(grid, start_pos):
    if grid[start_pos] >= 0: return start_pos
    q, visited = [start_pos], {start_pos}
    while q:
        (r, c) = q.pop(0)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and (nr, nc) not in visited:
                if grid[nr, nc] >= 0: return (nr, nc)
                q.append((nr, nc))
                visited.add((nr, nc))
    return None

def coords_to_grid(x, y, transform):
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col

def grid_to_coords(path, transform):
    coords = []
    for row, col in path:
        x = transform.c + (col + 0.5) * transform.a
        y = transform.f + (row + 0.5) * transform.e
        coords.append((x, y))
    return np.array(coords)


# --- Data Loading (Cached for performance) ---
@st.cache_data
def load_data():
    cost_grid = np.load("final_cost_grid.npy")
    points_gdf = gpd.read_file("iisertvm_map.gpkg", layer="points")
    buildings_gdf = gpd.read_file("iisertvm_map.gpkg", layer="buildings")
    
    points_gdf_proj = points_gdf.to_crs(epsg=3857)
    buildings_gdf_proj = buildings_gdf.to_crs(epsg=3857)
    
    GRID_WIDTH, GRID_HEIGHT = cost_grid.shape[1], cost_grid.shape[0]
    total_bounds = buildings_gdf_proj.total_bounds
    transform = from_bounds(total_bounds[0], total_bounds[1], total_bounds[2], total_bounds[3], GRID_WIDTH, GRID_HEIGHT)
    
    return cost_grid, points_gdf, points_gdf_proj, transform

cost_grid, points_gdf, points_gdf_proj, transform = load_data()


# --- Main App Interface ---
st.title("🛰️ AI Delivery Drone Mission Planner")
st.markdown("An A* pathfinding and route optimization agent for the IISER TVM campus.")

# --- User Controls in the Sidebar ---
with st.sidebar:
    st.header("Mission Controls")
    
    location_names = list(points_gdf['name'])
    
    start_location = st.selectbox("Select Start Location:", location_names)
    
    target_locations = st.multiselect(
        "Select Target Destinations:",
        options=location_names,
        help="Select one or more stops for the delivery route."
    )
    
    plan_button = st.button("Plan Mission", type="primary", use_container_width=True)

# (Keep all the code in app.py before this block the same)

# --- Mission Planning Logic ---
if plan_button:
    if not target_locations:
        st.error("Please select at least one target destination.")
    else:
        with st.spinner("Calculating all possible path lengths... This may take a moment."):
            # (The logic for calculating the distance_matrix is the same)
            start_idx = location_names.index(start_location)
            target_indices = [location_names.index(loc) for loc in target_locations]
            mission_indices = [start_idx] + target_indices
            mission_points = {idx: name for idx, name in enumerate(points_gdf['name']) if idx in mission_indices}
            
            distance_matrix = {}
            unique_mission_indices = list(set(mission_indices))
            for i, j in itertools.permutations(unique_mission_indices, 2):
                p_i = points_gdf_proj.geometry.iloc[i]
                p_j = points_gdf_proj.geometry.iloc[j]
                start_pos = find_nearest_walkable(cost_grid, coords_to_grid(p_i.x, p_i.y, transform))
                end_pos = find_nearest_walkable(cost_grid, coords_to_grid(p_j.x, p_j.y, transform))
                if start_pos and end_pos:
                    path = astar(cost_grid, start_pos, end_pos)
                    if path: distance_matrix[(i, j)] = len(path)

        with st.spinner("Optimizing route (Solving TSP)..."):
            # (The logic for finding the best_route is the same)
            best_route, min_distance = None, float('inf')
            for perm in itertools.permutations(target_indices):
                current_route, current_distance = [start_idx] + list(perm), 0
                for i in range(len(current_route) - 1):
                    try: current_distance += distance_matrix[(current_route[i], current_route[i+1])]
                    except KeyError: current_distance = float('inf'); break
                if current_distance < min_distance:
                    min_distance, best_route = current_distance, current_route
        
        st.header("Optimal Mission Plan")
        if best_route:
            route_names = [mission_points[i] for i in best_route]
            st.success(f"**Optimal Route:** {' -> '.join(route_names)}")
            st.info(f"**Total Path Length:** {min_distance} steps")

            # --- UPDATED VISUALIZATION BLOCK ---
            
            # 1. Reduce the figure size
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # (Plotting logic for path segments is the same)
            for i in range(len(best_route) - 1):
                p1_idx, p2_idx = best_route[i], best_route[i+1]
                p1_geom, p2_geom = points_gdf_proj.geometry.iloc[p1_idx], points_gdf_proj.geometry.iloc[p2_idx]
                start_pos = find_nearest_walkable(cost_grid, coords_to_grid(p1_geom.x, p1_geom.y, transform))
                end_pos = find_nearest_walkable(cost_grid, coords_to_grid(p2_geom.x, p2_geom.y, transform))
                
                path = astar(cost_grid, start_pos, end_pos)
                if path:
                    path_coords = grid_to_coords(path, transform)
                    ax.plot(path_coords[:, 0], path_coords[:, 1], color='cyan', linewidth=4, solid_capstyle='round', zorder=2)

            # (Plotting logic for stop markers is the same)
            for i, stop_idx in enumerate(best_route):
                 stop_geom = points_gdf_proj.geometry.iloc[stop_idx]
                 stop_pos_grid = find_nearest_walkable(cost_grid, coords_to_grid(stop_geom.x, stop_geom.y, transform))
                 stop_pos_coords = grid_to_coords([stop_pos_grid], transform)[0]
                 ax.scatter(stop_pos_coords[0], stop_pos_coords[1], s=300, c='magenta', zorder=3, edgecolor='black')
                 ax.text(stop_pos_coords[0], stop_pos_coords[1], f"{i+1}", fontsize=16, color='white', fontweight='bold', ha='center', va='center',path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='black')])
            
            # 2. Add interpolation for a "softer" image
            cx.add_basemap(
                ax,
                source=cx.providers.Esri.WorldImagery,
                crs=points_gdf_proj.crs.to_string(),
                interpolation='spline16' 
            )
            
            # 3. Enforce the correct aspect ratio
            ax.set_aspect('equal')
            
            ax.set_axis_off()
            st.pyplot(fig)
            
        else:
            st.error("Could not find a valid route that visits all selected targets.")

# (The else block at the end of the file is the same)
else:
    st.info("Select a start location and one or more destinations in the sidebar, then click 'Plan Mission'.")
