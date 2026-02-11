import torch


def swap(solution: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Swap two nodes in the solution.

    Args:
        solution: Tensor [batch, num_nodes, 1]
        indices: Tensor [batch, 2] containing swap indices

    Returns:
        New solution with swapped nodes
    """
    sol = solution.clone()
    batch_idx = torch.arange(sol.size(0), device=sol.device)
    idx1, idx2 = indices[:, 0], indices[:, 1]

    temp = sol[batch_idx, idx1]
    sol[batch_idx, idx1] = sol[batch_idx, idx2]
    sol[batch_idx, idx2] = temp

    return sol


def two_opt(x: torch.Tensor, a: torch.Tensor):
    """
    Perform 2-opt move by reversing segment between indices.

    Args:
        x: Tensor [batch, num_nodes, 1]
        a: Tensor [batch, 2] containing segment endpoints

    Returns:
        Solution with reversed segment
    """
    # Two-opt moves invert a section of a tour. If we cut a tour into
    # segments a and b then we can choose to invert either a or b. Due
    # to the linear representation of a tour, we choose always to invert
    # the segment that is stored contiguously.
    left = torch.minimum(a[:, 0], a[:, 1])
    right = torch.maximum(a[:, 0], a[:, 1])
    batch_size, dim = x.size(0), x.size(1)
    ones = torch.ones((batch_size, 1), dtype=torch.long, device=x.device)
    fidx = torch.arange(dim, device=x.device) * ones
    # Reversed indices
    offset = left + right - 1
    ridx = torch.arange(0, -dim, -1, device=x.device) + offset[:, None]
    # Set flipped section to all True
    flip = torch.ge(fidx, left[:, None]) * torch.lt(fidx, right[:, None])
    # Set indices to replace flipped section with
    idx = (~flip) * fidx + flip * ridx
    # Perform 2-opt move
    return torch.gather(x, 1, idx.unsqueeze(-1))


def two_opt_star(
    x: torch.Tensor, a: torch.Tensor, segment_ids: torch.Tensor
) -> torch.Tensor:
    """
    Performs 2-Opt* (Inter-Route Tail Swap).
    Swaps the tails (all nodes after the selected index) between two routes.
    """
    batch_size, seq_len = x.size(0), x.size(1)

    # Extract indices u (source) and v (target)
    u_idx = a[:, 0]  # [B]
    v_idx = a[:, 1]  # [B]

    # We need to build the new sequence.
    # Constructing a mask to reorder is tricky because tail lengths vary.
    # We will build the new tensor by gathering.

    range_idx = (
        torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
    )

    # Identify segments
    u_seg = segment_ids.gather(1, u_idx.unsqueeze(1))
    v_seg = segment_ids.gather(1, v_idx.unsqueeze(1))

    current_seg_ids = segment_ids

    # Define masks for the four parts involved in the swap:
    # Route U Head: segment == u_seg AND index <= u
    # Route U Tail: segment == u_seg AND index > u
    # Route V Head: segment == v_seg AND index <= v
    # Route V Tail: segment == v_seg AND index > v

    is_u_seg = current_seg_ids == u_seg
    is_v_seg = current_seg_ids == v_seg

    u_head_mask = is_u_seg & (range_idx <= u_idx.unsqueeze(1))
    u_tail_mask = is_u_seg & (range_idx > u_idx.unsqueeze(1))
    v_head_mask = is_v_seg & (range_idx <= v_idx.unsqueeze(1))
    v_tail_mask = is_v_seg & (range_idx > v_idx.unsqueeze(1))

    # Everything else stays in place
    keep_mask = ~(is_u_seg | is_v_seg)

    # We need to construct the new order.
    # The new Route U will contain: U_Head -> V_Tail
    # The new Route V will contain: V_Head -> U_Tail
    # However, simply masking isn't enough; we need to shift indices because
    # the lengths of U_Tail and V_Tail might differ.

    # Solution: Sort/argsort based on a constructed priority key.
    # 1. Keepers: Original Index
    # 2. U_Head: Original Index
    # 3. V_Tail: Moves to where U_Tail was.
    # 4. V_Head: Original Index
    # 5. U_Tail: Moves to where V_Tail was.

    # To simplify implementation in a rigid tensor:
    # We assign a floating point 'position' to every node.
    # - Standard nodes: pos = index
    # - V_Tail nodes: Should come immediately after U_Head.
    #   Let's map them to range (u_idx, u_idx + 1).
    # - U_Tail nodes: Should come immediately after V_Head.
    #   Let's map them to range (v_idx, v_idx + 1).

    new_pos = range_idx.float()

    # Compress V_Tail to fit after U
    # We spread the indices of V_tail between u and u+1
    # Normalized offset = (index - v_idx) / (seq_len)
    v_tail_indices = torch.where(v_tail_mask, range_idx, torch.zeros_like(range_idx))
    # We need them sorted relative to each other, but placed after u
    # New Pos = u_idx + 0.1 + (original_index * small_epsilon)
    new_pos = torch.where(
        v_tail_mask,
        u_idx.unsqueeze(1).float() + 0.1 + (range_idx.float() * 1e-5),
        new_pos,
    )

    # Compress U_Tail to fit after V
    new_pos = torch.where(
        u_tail_mask,
        v_idx.unsqueeze(1).float() + 0.1 + (range_idx.float() * 1e-5),
        new_pos,
    )

    # Get the sort order to rearrange the tensor
    _, sort_indices = torch.sort(new_pos, dim=1)

    # Gather the new solution
    return torch.gather(x, 1, sort_indices.unsqueeze(-1))


def insertion(solution: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Performs the insertion operator for CVRP.
    Moves the node at `source_idx` to the position immediately AFTER `target_idx`.

    Args:
        solution: Current solution tensor of shape [batch_size, num_nodes, feature_dim]
                  or [batch_size, num_nodes].
        action: Tensor of shape [batch_size, 2] containing [source_idx, target_idx]
                for each batch instance.

    Returns:
        The modified solution tensor with the node moved.
    """
    # 1. Unpack actions
    # src_idx: The index of the node we are moving
    # tgt_idx: The index of the node we are inserting AFTER
    src_idx = action[:, 0]
    tgt_idx = action[:, 1]

    batch_size, num_nodes = solution.shape[:2]
    device = solution.device

    # 2. Create base "scores" representing the current order [0, 1, 2, ..., N-1]
    # Shape: [batch_size, num_nodes]
    # Example: [0.0, 1.0, 2.0, 3.0, 4.0]
    scores = torch.arange(num_nodes, device=device, dtype=torch.float32)
    scores = scores.unsqueeze(0).expand(batch_size, -1).clone()

    # 3. Define the new position for the source node
    # We want the Source to appear immediately AFTER the Target.
    # If Target has score X, we give Source score X + 0.5.
    # This places it mathematically between Target (X) and the next node (X + 1).
    target_scores = tgt_idx.float() + 0.5

    # 4. Update the scores
    # We scatter the new score into the position of the source node.
    # scores[b, src_idx[b]] = target_scores[b]
    scores.scatter_(1, src_idx.unsqueeze(1), target_scores.unsqueeze(1))

    # 5. Determine the new order
    # Sort the scores to get the new indices.
    # stable=True helps stability, though floating point logic here is robust.
    _, new_order = torch.sort(scores, dim=1)

    # 6. Apply the reordering to the solution
    if solution.dim() == 3:
        # If solution has features (e.g., coordinates), expand indices to match
        # Shape: [batch, num_nodes, features]
        idx_expanded = new_order.unsqueeze(-1).expand_as(solution)
        new_solution = torch.gather(solution, 1, idx_expanded)
    else:
        # If solution is just a list of node IDs
        # Shape: [batch, num_nodes]
        new_solution = torch.gather(solution, 1, new_order)

    return new_solution


# def insertion(solution: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
#     """
#     Move a node to a new position in the solution.

#     Args:
#         solution: Tensor [batch, route_length, 1]
#         indices: Tensor [batch, 2] -> (node_position, new_position)

#     Returns:
#         Modified solution tensor
#     """
#     batch_size, route_length, _ = solution.shape
#     device = solution.device

#     # Remove last dimension and ensure proper type
#     solution = solution.squeeze(-1)
#     node_pos = indices[:, 0]
#     new_pos = indices[:, 1].clamp(0, route_length - 1)

#     # Create batch indices [0, 1, ..., batch_size-1]
#     batch_idx = torch.arange(batch_size, device=device)

#     # Create mask for all elements except the ones to move
#     mask = torch.ones_like(solution, dtype=torch.bool)
#     mask[batch_idx, node_pos] = False

#     # Get remaining nodes after removing the moved ones
#     remaining_nodes = solution[mask].view(batch_size, route_length - 1)

#     # Create new solution tensor with proper dtype
#     new_solution = torch.zeros(
#         batch_size, route_length, dtype=solution.dtype, device=device
#     )

#     # Create position ranges
#     pos_range = torch.arange(route_length, device=device).expand(batch_size, -1)

#     # Create position ranges for remaining_nodes
#     # C'est la partie la plus complexe de votre implémentation
#     pos_range_rem = torch.arange(route_length - 1, device=device).expand(batch_size, -1)

#     # Build the new solution by scattering
#     # For positions before new_pos, take from remaining_nodes
#     mask_new_before = pos_range < new_pos.unsqueeze(1)
#     mask_rem_before = pos_range_rem < new_pos.unsqueeze(1)

#     # Correction pour gérer les tailles de masques potentiellement différentes
#     # On s'assure qu'on assigne le bon nombre d'éléments
#     # Le nombre d'éléments 'True' dans mask_new_before[i] est new_pos[i]
#     # Le nombre d'éléments 'True' dans mask_rem_before[i] est new_pos[i]
#     # Votre logique originale était correcte car les masques ont le même nombre de True
#     new_solution[mask_new_before] = remaining_nodes[mask_rem_before]

#     # For positions after new_pos, take from remaining_nodes offset by 1
#     mask_new_after = pos_range > new_pos.unsqueeze(1)
#     mask_rem_after = pos_range_rem >= new_pos.unsqueeze(1)

#     new_solution[mask_new_after] = remaining_nodes[mask_rem_after]

#     # Insert the moved nodes at their new positions
#     new_solution[batch_idx, new_pos] = solution[batch_idx, node_pos]

#     return new_solution.unsqueeze(-1)
