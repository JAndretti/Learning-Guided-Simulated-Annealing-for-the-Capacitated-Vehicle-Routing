import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

# 1. Load your data
coords = torch.load("example/data/coordinates.pt")
states = torch.load("example/data/states.pt")
costs = torch.load("example/data/costs.pt")

# Move to CPU and convert to numpy
coords_np = coords[0].cpu().numpy()

# 2. Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the depot (red square) and customers (blue dots)
depot = coords_np[0]
ax.scatter(depot[0], depot[1], c="red", marker="s", s=100, label="Depot", zorder=3)
customers = coords_np[1:]
ax.scatter(
    customers[:, 0],
    customers[:, 1],
    c="blue",
    marker="o",
    s=20,
    label="Customers",
    zorder=2,
)

# Use the 'tab10' colormap which has 10 highly distinct, bold colors.
max_vehicles = 101  # Set higher to support up to 101 routes
cmap = plt.get_cmap("tab10")
route_lines = []
for i in range(max_vehicles):
    # Using modulo (%) ensures that if you accidentally get more routes,
    # it just loops back to the first color instead of crashing.
    (line,) = ax.plot(
        [], [], "-", linewidth=2.5, alpha=0.9, zorder=1, color=cmap(i % 10)
    )
    route_lines.append(line)

ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)

# Keep axis limits fixed
ax.set_xlim(coords_np[:, 0].min() - 0.05, coords_np[:, 0].max() + 0.05)
ax.set_ylim(coords_np[:, 1].min() - 0.05, coords_np[:, 1].max() + 0.05)


# 3. Define the update function
def update(frame):
    current_state = states[frame][0, :, 0].cpu().numpy()

    # Find all indices where the sequence is at the depot (0)
    zero_indices = np.where(current_state == 0)[0]

    route_idx = 0
    # NEW: Keep a dedicated counter for active routes
    active_routes = 0

    # Loop through the zeros to extract the routes between them
    for i in range(len(zero_indices) - 1):
        start_idx = zero_indices[i]
        end_idx = zero_indices[i + 1]

        # Check if it's a real route
        if end_idx - start_idx > 1:
            route_nodes = current_state[start_idx : end_idx + 1]
            route_coords = coords_np[route_nodes]

            # Update line data
            route_lines[route_idx].set_data(route_coords[:, 0], route_coords[:, 1])
            route_idx += 1
            active_routes += 1

    # Hide any unused lines
    for i in range(route_idx, max_vehicles):
        route_lines[i].set_data([], [])

    # Extract current cost and calculate the best
    current_cost = costs[frame].item()
    best_step = torch.argmin(torch.tensor(costs)[: frame + 1]).item()
    best_cost = costs[best_step].item()

    # NEW: Add the active_routes count to the title
    title_text = (
        f"Step: {frame} | Routes: {active_routes} | Current Cost: {current_cost:.2f}\n"
        f"Best Cost: {best_cost:.2f} (Found at Step {best_step})"
    )
    ax.set_title(title_text, fontsize=12, fontweight="bold")

    return route_lines


# 4. Create the animation
ani = FuncAnimation(fig, update, frames=len(states), interval=50, blit=False)

# 5. Show or save
plt.show()

# To save it as a GIF (uncomment below):
# ani.save("example/plots/cvrp_evolution.gif", writer="pillow", fps=20)

# To save it as an MP4 video (requires ffmpeg installed, uncomment below):
# ani.save('cvrp_evolution.mp4', writer='ffmpeg', fps=20)
