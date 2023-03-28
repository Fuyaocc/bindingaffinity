import collections

import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange, repeat

from prodesign.common import residue_constants
from prodesign.utils import default, exists

def l2_norm(v, dim=-1, epsilon=1e-12):
    return v / torch.clamp(torch.linalg.norm(v, dim=dim, keepdim=True), min=epsilon)

def masked_mean(mask, value, epsilon=1e-10):
    return torch.sum(mask*value) / (epsilon + torch.sum(mask))

def batched_gather(params, indices):
    b, device = indices.shape[0], indices.device
    params = torch.from_numpy(params).to(device)
    params = repeat(params, 'n ... -> b n ...', b=b)
    c = len(params.shape) - len(indices.shape)
    assert c >= 0
    ext = list(map(chr, range(ord('o'), ord('o') + c)))
    kwargs = dict(zip(ext, params.shape[-c:]))
    ext = ' '.join(ext)
    return torch.gather(params, 1, repeat(indices, f'b n ... -> b n ... {ext}', **kwargs))

"""
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)

def standardize_quaternion(quaternions):
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a, b):
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a, b):
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def lddt(pred_points, true_points, points_mask, cutoff=15.):
    """Computes the lddt score for a batch of coordinates.
        https://academic.oup.com/bioinformatics/article/29/21/2722/195896
        Inputs: 
        * pred_coords: (b, l, d) array of predicted 3D points.
        * true_points: (b, l, d) array of true 3D points.
        * points_mask : (b, l) binary-valued array. 1 for points that exist in
            the true points
        * cutoff: maximum inclusion radius in reference struct.
        Outputs:
        * (b, l) lddt scores ranging between 0 and 1
    """
    assert len(pred_points.shape) == 3 and pred_points.shape[-1] == 3
    assert len(true_points.shape) == 3 and true_points.shape[-1] == 3

    eps = 1e-10

    # Compute true and predicted distance matrices. 
    pred_cdist = torch.cdist(pred_points, pred_points, p=2)
    true_cdist = torch.cdist(true_points, true_points, p=2)

    cdist_to_score = ((true_cdist < cutoff) *
            (rearrange(points_mask, 'b i -> b i ()')*rearrange(points_mask, 'b j -> b () j')) *
            (1.0 - torch.eye(true_cdist.shape[1], device=points_mask.device)))  # Exclude self-interaction

    # Shift unscored distances to be far away
    dist_l1 = torch.abs(true_cdist - pred_cdist)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * sum([dist_l1 < t for t in (0.5, 1.0, 2.0, 4.0)])

    # Normalize over the appropriate axes.
    return (torch.sum(cdist_to_score * score, dim=-1) + eps)/(torch.sum(cdist_to_score, dim=-1) + eps)

def plddt(logits):
    """Compute per-residue pLDDT from logits
    """
    device = logits.device if hasattr(logits, 'device') else None
    # Shape (b, l, c)
    b, c = logits.shape[0], logits.shape[-1]
    width = 1.0 / c
    centers = torch.arange(start=0.5*width, end=1.0, step=width, device=device)
    probs = F.softmax(logits, dim=-1)
    return torch.einsum('c,... c -> ...', centers, probs)

Rigids = collections.namedtuple('Rigids', ['rotations', 'translations'])

def rigids_from_3x3(points, indices=None, epsilon=1e-6):
    """Create rigids from 3 points.
    This creates a set of rigid transformations from 3 points by Gram Schmidt
    orthogonalization.
    """
    indices = default(indices, (0, 1, 2))
    # Shape (b, l, 3, 3)
    assert points.shape[-1] == 3
    assert points.shape[-2] >= 3
    v1 = points[...,indices[0],:] - points[...,indices[1],:]
    v2 = points[...,indices[2],:] - points[...,indices[1],:]

    
    e1 = l2_norm(v1, epsilon=epsilon)
    c = torch.sum(e1 * v2, dim=-1, keepdim=True)
    u2 = v2 - e1*c
    e2 = l2_norm(u2, epsilon=epsilon)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack((e1, e2, e3), dim=-1)
    t = points[...,indices[1],:]

    return R, t

def rigids_from_4x4(m):
    """Create rigids from 4x4 array
    """
    # Shape (..., 4, 4)
    assert m.shape[-2:] == (4, 4)
    return m[...,:3,:3], m[...,:3,3]

def angles_from_positions(aatypes, coords, coord_mask):
    prev_coords, prev_mask = coords[...,:-1,:,:], coord_mask[...,:-1,:]
    this_coords, this_mask = coords[...,1:,:,:], coord_mask[...,1:,:]

    #omega_points = torch.stack((prev_coords[...,]))
    # (N, 7, 4, 14)
    torsion_atom14_idx = F.one_hot(
            batched_gather(residue_constants.chi_angles_atom14_indices, aatypes),
            residue_constants.atom14_type_num)
    torsion_atom14_exists = batched_gather(residue_constants.chi_angles_atom14_exists, aatypes)

    # (N, 7, 4, 3)
    torsion_points = torch.einsum('... n d,... g m n -> ... g m d', coords,
            torsion_atom14_idx.float())
    # (N, 7, 4)
    torsion_point_mask = torch.einsum('... n,... g m n -> ... g m', coord_mask.float(),
            torsion_atom14_idx.float())

    # fix omega, phi angles
    for i in range(torsion_points.shape[-4] - 1, 0, -1):
        torsion_points[...,i,0,:2,:] = torsion_points[...,i-1,0,:2,:]  # omega
        torsion_point_mask[...,i,0,:2] = torsion_point_mask[...,i-1,0,:2]

        torsion_points[...,i,1,:1,:] = torsion_points[...,i-1,1,:1,:]  # phi
        torsion_point_mask[...,i,1,:1] = torsion_point_mask[...,i-1,1,:1]

    torsion_points[...,0,0,:2,:] = 0  # omega
    torsion_point_mask[...,0,0,:2] = 0
    torsion_points[...,0,1,:1,:] = 0  # phi
    torsion_point_mask[...,0,1,:1] = 0

    # Create a frame from the first three atoms:
    # First atom: point on x-y-plane
    # Second atom: point on negative x-axis
    # Third atom: origin
    torsion_frames = rigids_from_3x3(torsion_points, indices=(2, 1, 0))

    # Compute the position of the forth atom in this frame (y and z coordinate
    # define the chi angle)
    def to_local(rotations, translations, points):
        rotations = rearrange(rotations, '... h w -> ... w h')
        translations = -torch.einsum('... w,... h w -> ... h', translations, rotations)
        return torch.einsum('... w,... h w -> ... h', points, rotations) + translations
    def to_angles(rotations, translations, torsion_points_4):
        local_points = to_local(rotations, translations, torsion_points_4)
        angles_sin_cos = torch.stack((local_points[...,2], local_points[...,1]), dim=-1)
        return angles_sin_cos / torch.sqrt(1e-8 +
                torch.sum(angles_sin_cos**2, dim=-1, keepdim=True))

    torsion_angles = to_angles(*torsion_frames, torsion_points[...,3,:])
    torsion_angles_mask = torch.prod(torsion_point_mask, dim=-1) * torsion_atom14_exists

    # Mirror psi, because we computed it from the Oxygen-atom.
    torsion_angles *= rearrange(
            torch.as_tensor([1.,1.,-1.,1.,1.,1.,1.], device=torsion_angles.device),
            'g -> g ()')

    # Create alternative angles for ambiguous atom names.
    chi_is_ambiguous = batched_gather(np.asarray(residue_constants.chi_pi_periodic), aatypes)
    mirror_ambiguous = torch.cat((
            torch.ones(aatypes.shape + (3,), device=aatypes.device),
            1 - 2*chi_is_ambiguous), dim=-1)
    torsion_angles_alt = torsion_angles * mirror_ambiguous[...,None]

    return dict(torsion_angles=torsion_angles,
            torsion_angles_mask=torsion_angles_mask,
            torsion_angles_alt=torsion_angles_alt)

def rigids_from_positions(aatypes, coords, coord_mask):
    # (N, 8, 3, 14)
    group_atom14_idx = F.one_hot(
            batched_gather(residue_constants.restype_rigid_group_atom14_idx, aatypes),
            residue_constants.atom14_type_num)
    # Compute a mask whether the group exists.
    # (N, 8)
    group_exists = batched_gather(residue_constants.restype_rigid_group_mask, aatypes)
    if not exists(coords):
        return dict(atom_affine_exists=group_exists)

    group_points = torch.einsum('... n d,... g m n -> ... g m d', coords,
            group_atom14_idx.float())
    group_point_mask = torch.einsum('... n,... g m n', coord_mask.float(),
            group_atom14_idx.float())

    # Compute a mask whether ground truth exists for the group
    group_mask = torch.all(group_point_mask > 0, dim=-1) * group_exists

    # Compute the Rigids.
    group_affine = rigids_from_3x3(group_points)

    # The frames for ambiguous rigid groups are just rotated by 180 degree around
    # the x-axis. The ambiguous group is always the last chi-group.
    restype_rigid_group_is_ambiguous = np.zeros([21, 8], dtype=np.float32)
    restype_rigid_group_rotations = np.tile(np.eye(3, dtype=np.float32), [21, 8, 1, 1])

    for resname, _ in residue_constants.residue_atom_renaming_swaps.items():
        restype = residue_constants.restype_order[
                residue_constants.restype_3to1[resname]]
        chi_idx = int(sum(residue_constants.chi_angles_mask[restype]) - 1)
        restype_rigid_group_is_ambiguous[restype, chi_idx + 4] = 1
        restype_rigid_group_rotations[restype, chi_idx + 4, 1, 1] = -1
        restype_rigid_group_rotations[restype, chi_idx + 4, 2, 2] = -1

    group_is_ambiguous = batched_gather(restype_rigid_group_is_ambiguous, aatypes)
    group_ambiguous_rotations = batched_gather(restype_rigid_group_rotations, aatypes)

    # Create the alternative ground truth frames.
    group_affine_alt = rigids_rotate(group_affine, group_ambiguous_rotations)

    return dict(atom_affine=group_affine,
            atom_affine_exists=group_exists,
            atom_affine_mask=group_mask,
            atom_affine_is_ambiguous=group_is_ambiguous,
            atom_affine_alt=group_affine_alt)

def rigids_to_positions(frames, aatypes):
    # Shape ((b, l, 8, 3, 3), (b, l, 8, 3))
    rotations, translations = frames

    # Shape (b, l, 14)
    group_idx = batched_gather(
        residue_constants.restype_atom14_to_rigid_group, aatypes)
    # Shape (b, l, 14, 8)
    group_mask = F.one_hot(group_idx, num_classes=8).float()

    rotations = torch.einsum('b l m n,b l n h w->b l m h w', group_mask, rotations)
    translations = torch.einsum('b l m n,b l n h->b l m h', group_mask, translations)

    # Gather the literature atom positions for each residue.
    # Shape (b, l, 14, 3)
    group_pos = batched_gather(
        residue_constants.restype_atom14_rigid_group_positions, aatypes)

    # Transform each atom from it's local frame to the global frame.
    positions = torch.einsum('... w,... h w -> ... h', group_pos, rotations) + translations

    # Mask out non-existing atoms.
    mask = batched_gather(
        residue_constants.restype_atom14_mask, aatypes)
    return positions*rearrange(mask, '... d->... d ()')

def rigids_slice(frames, start=0, end=None):
    rotations, translations = frames
    return rotations[...,start:end,:,:], translations[...,start:end,:]

def rigids_rearrange(frames, ops):
    rotations, translations = frames
    return rearrange(rotations, ops), rearrange(translations, ops)

def rigids_multiply(a, b):
    rots_a, trans_a = a
    rots_b, trans_b = b
    rotations = torch.einsum('... h d,... d w -> ... h w', rots_a, rots_b)
    translations = torch.einsum('... w,... h w -> ... h', trans_b, rots_a) + trans_a
    return rotations, translations

def rigids_rotate(frames, mat3x3):
    rotations, translations = frames
    rotations = torch.einsum('... h d, ... d w -> ... h w', rotations, mat3x3)
    return rotations, translations

def rigids_from_angles(aatypes, backb_frames, angles):
    """Create rigids from torsion angles
    """
    # Shape (b, l, 3, 3), (b, l, 3)
    backb_rotations, backb_trans = backb_frames
    assert backb_rotations.shape[-2:] == (3, 3) and backb_trans.shape[-1] == 3
    # Shape (b, l)
    assert aatypes.shape == backb_rotations.shape[:-2]
    # Shape (b, l, n, 2) s.t. n <= 7
    assert angles.shape[-1] == 2 and angles.shape[-2] <= 7
    assert angles.shape[:-2] == aatypes.shape
    
    b, l, n = angles.shape[:3]

    # Gather the default frames for all rigids (b, l, 8, 3, 3), (b, l, 8, 3)
    m = batched_gather(residue_constants.restype_rigid_group_default_frame, aatypes)
    default_frames = rigids_slice(rigids_from_4x4(m), 0, n+1)

    #
    # Create the rotation matrices according to the given angles
    #

    # Insert zero rotation for backbone group.
    # Shape (b, l, n+1, 2)
    angles = torch.cat((
            rearrange(torch.stack((torch.zeros_like(aatypes), torch.ones_like(aatypes)), dim=-1), 'b l r -> b l () r'),
            angles), dim=-2)
    sin_angles, cos_angles = torch.unbind(angles, dim=-1)
    zeros, ones = torch.zeros_like(sin_angles), torch.ones_like(cos_angles)
    # Shape (b, l, n+1, 3, 3)
    rotations = torch.stack((
            ones, zeros, zeros,
            zeros, cos_angles, -sin_angles,
            zeros, sin_angles, cos_angles), dim=-1)
    rotations = rearrange(rotations, '... (h w) -> ... h w', h=3, w=3)

    # Apply rotations to the frames.
    atom_frames = rigids_rotate(default_frames, rotations)

    # \chi_2, \chi_3, and \chi_4 frames do not transform to the backbone frame but to
    # the previous frame. So chain them up accordingly.
    def to_prev_frames(frames, idx):
        rotations, translations = frames

        assert rotations.device == translations.device
        assert rotations.shape[:-2] == translations.shape[:-1]

        ri, ti = rigids_multiply(
                (rotations[...,idx-1:idx,:,:], translations[...,idx-1:idx,:]),
                (rotations[...,idx:idx+1,:,:], translations[...,idx:idx+1,:]))
        return torch.cat((rotations[...,:idx,:,:], ri, rotations[...,idx+1:,:,:]), dim=-3), torch.cat((translations[...,:idx,:], ti, translations[...,idx+1:,:]), dim=-2)

    for i in range(5, n+1):
        atom_frames = to_prev_frames(atom_frames, i)

    return rigids_multiply(rigids_rearrange(backb_frames, 'b l ... -> b l () ...'), atom_frames)

def fape(pred_frames, true_frames, frames_mask, pred_points, true_points, points_mask, clamp_distance=None, epsilon=1e-8):
    """ FAPE(Frame Aligined Point Error) - Measure point error under different alignments
    """
    # Shape (b, l, 3, 3), (b, l, 3)
    pred_rotations, pred_trans = pred_frames
    assert pred_rotations.shape[-2:] == (3, 3) and pred_trans.shape[-1] == 3
    # Shape (b, l, 3, 3), (b, l, 3)
    true_rotations, true_trans = true_frames
    assert true_rotations.shape[-2:] == (3, 3) and true_trans.shape[-1] == 3
    ## Shape (b, l)
    #assert frames_mask.shape[:2] == points_mask.shape[:2]
    # Shape (b, n, 3)
    assert pred_points.shape[-1] == 3 and true_points.shape[-1] == 3
    # Shape (b, n)
    assert pred_points.shape[:-1] == points_mask.shape

    def to_local(rotations, translations, points):
        # inverse frames
        rotations = rearrange(rotations, '... h w -> ... w h')
        translations = -torch.einsum('... w,... h w -> ... h', translations, rotations)
        return torch.einsum('... j w,... i h w -> ... i j h', points, rotations) + rearrange(translations, '... i h -> ... i () h')
    
    pred_xij = to_local(pred_rotations, pred_trans, pred_points)
    true_xij = to_local(true_rotations, true_trans, true_points)

    # Shape (b, l, l, n)
    dij = torch.sqrt(
            torch.sum((pred_xij - true_xij)**2, dim=-1) + epsilon)
    if clamp_distance:
        dij = torch.clip(dij, 0, clamp_distance)
    dij_mask = rearrange(frames_mask, '... i -> ... i ()') * rearrange(points_mask, '... j -> ... () j')

    return torch.sum(dij * dij_mask) / (epsilon + torch.sum(dij_mask))

def between_ca_ca_distance_loss(pred_points, points_mask, residue_index, tau=1.5, epsilon=1e-6):
    assert pred_points.shape[-1] == 3
    assert pred_points.shape[:-1] == points_mask.shape
    assert points_mask.shape[:-1] == residue_index.shape

    ca_idx = residue_constants.atom_order['CA']

    this_ca_point = pred_points[...,:-1,ca_idx,:]
    this_ca_mask = points_mask[...,:-1,ca_idx]
    next_ca_point = pred_points[...,1:,ca_idx,:]
    next_ca_mask = points_mask[...,1:,ca_idx]
    no_gap_mask = ((residue_index[...,1:] - residue_index[...,:-1]) == 1)

    ca_ca_distance = torch.sqrt(
            epsilon + torch.sum((this_ca_point - next_ca_point)**2, dim=-1))
    violations = torch.gt(ca_ca_distance - residue_constants.ca_ca, tau)
    mask = this_ca_mask * next_ca_mask * no_gap_mask
    return masked_mean(mask=mask, value=violations, epsilon=epsilon)

def between_residue_bond_loss(pred_points, points_mask, residue_index, aatypes, tau=12.0, epsilon=1e-6, loss_only=False):
    assert pred_points.shape[-1] == 3
    assert pred_points.shape[:-1] == points_mask.shape
    assert points_mask.shape[:-1] == residue_index.shape
    assert aatypes.shape == residue_index.shape

    n_idx = residue_constants.atom_order['N']
    ca_idx = residue_constants.atom_order['CA']
    c_idx = residue_constants.atom_order['C']

    # Get the positions of the relevant backbone atoms.
    this_ca_point = pred_points[...,:-1,ca_idx,:]
    this_ca_mask = points_mask[...,:-1,ca_idx]
    this_c_point = pred_points[...,:-1,c_idx,:]
    this_c_mask = points_mask[...,:-1,c_idx]
    next_n_point = pred_points[...,1:,n_idx,:]
    next_n_mask = points_mask[...,1:,n_idx]
    next_ca_point = pred_points[...,1:,ca_idx,:]
    next_ca_mask = points_mask[...,1:,ca_idx]
    no_gap_mask = ((residue_index[...,1:] - residue_index[...,:-1]) == 1)


    # Compute bond length
    c_n_bond_length = torch.sqrt(
            epsilon + torch.sum((this_c_point - next_n_point)**2, dim=-1))
    ca_c_bond_length = torch.sqrt(
            epsilon + torch.sum((this_ca_point - this_c_point)**2, dim=-1))
    n_ca_bond_length = torch.sqrt(
            epsilon + torch.sum((next_n_point - next_ca_point)**2, dim=-1))

    # Compute loss for the C--N bond.
    # The C-N bond to proline has slightly different length because of the ring.
    def bond_length_loss(pred_length, gt, mask):
        gt_length, gt_stddev = gt
        length_errors = torch.sqrt(
                epsilon + (pred_length - gt_length)**2)
        length_loss = F.relu(length_errors - tau*gt_stddev)
        return (length_loss,
                masked_mean(mask=mask, value=length_loss, epsilon=epsilon),
                mask * (length_errors > tau*gt_stddev))

    next_is_proline = (
            aatypes[...,1:] == residue_constants.resname_to_idx['PRO'])
    c_n_bond_labels = ((~next_is_proline)*residue_constants.between_res_bond_length_c_n[0] +
            next_is_proline*residue_constants.between_res_bond_length_c_n[1])
    c_n_bond_stddev = ((~next_is_proline)*residue_constants.between_res_bond_length_stddev_c_n[0] +
            next_is_proline*residue_constants.between_res_bond_length_stddev_c_n[1])
    c_n_bond_errors, c_n_loss, c_n_violation_mask = bond_length_loss(c_n_bond_length,
            (c_n_bond_labels, c_n_bond_stddev), this_c_mask * next_n_mask * no_gap_mask)

    # Compute loss for the angles.
    c_ca_unit_vec = (this_ca_point - this_c_point) / ca_c_bond_length[...,None]
    c_n_unit_vec = (next_n_point - this_c_point) / c_n_bond_length[...,None]
    n_ca_unit_vec = (next_ca_point - next_n_point) / n_ca_bond_length[...,None]

    def bond_angle_loss(x, y, gt, mask):
        gt_angle, gt_stddev = gt
        pred_angle = torch.sum(x*y, dim=-1)
        angle_errors = torch.sqrt(
                epsilon + (pred_angle - gt_angle)**2)
        angle_loss = F.relu(angle_errors - tau*gt_stddev)
        return (angle_loss,
                masked_mean(mask=mask, value=angle_loss, epsilon=epsilon),
                mask * (angle_errors > tau*gt_stddev))

    ca_c_n_erros, ca_c_n_loss, ca_c_n_violation_mask = bond_angle_loss(c_ca_unit_vec, c_n_unit_vec,
            residue_constants.between_res_cos_angles_ca_c_n,
            this_ca_mask*this_c_mask*next_n_mask*no_gap_mask)
    c_n_ca_errors, c_n_ca_loss, c_n_ca_violation_mask = bond_angle_loss(-c_n_unit_vec, n_ca_unit_vec,
            residue_constants.between_res_cos_angles_c_n_ca,
            this_c_mask*next_n_mask*next_ca_mask*no_gap_mask)

    if loss_only:
        return dict(c_n_loss=c_n_loss,
                ca_c_n_loss=ca_c_n_loss,
                c_n_ca_loss=c_n_ca_loss)

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    per_residue_violation = (c_n_bond_errors + ca_c_n_erros + c_n_ca_errors)
    per_residue_violation = 0.5 * (F.pad(per_residue_violation, (0, 1)) +
            F.pad(per_residue_violation, (1, 0)))

    # Compute hard violations.
    per_residue_violation_mask = torch.amax(torch.stack((c_n_violation_mask,
            ca_c_n_violation_mask, c_n_ca_violation_mask), dim=0), dim=0)
    per_residue_violation_mask = torch.maximum(F.pad(per_residue_violation_mask, (0, 1)),
            F.pad(per_residue_violation_mask, (1, 0)))

    return dict(c_n_loss=c_n_loss,
                ca_c_n_loss=ca_c_n_loss,
                c_n_ca_loss=c_n_ca_loss,
                per_residue_violation=per_residue_violation,
                per_residue_violation_mask=per_residue_violation_mask)

def between_residue_clash_loss(pred_points, points_mask, residue_index, aatypes, tau=1.5, epsilon=1e-6, loss_only=False):
    """Loss to penalize steric clashes between residues"""
    assert pred_points.shape[-1] == 3
    assert pred_points.shape[:-1] == points_mask.shape
    assert points_mask.shape[:-1] == residue_index.shape
    assert aatypes.shape == residue_index.shape

    atom_radius = batched_gather(
            residue_constants.atom14_van_der_waals_radius,
            aatypes)
    atom_radius = atom_radius[...,:points_mask.shape[-1]]
    assert atom_radius.shape == points_mask.shape

    # Create the distance matrix
    dists = torch.sqrt(epsilon + torch.sum(
            (rearrange(pred_points,
                    '... i m d -> ... i () m () d') - 
             rearrange(pred_points,
                    '... j n d -> ... () j () n d'))**2, dim=-1))

    # Create the mask for valid distances.
    dist_mask = (
                rearrange(points_mask, '... i m -> ... i () m ()') *
                rearrange(points_mask, '... j n -> ... () j () n'))
    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dist_mask *= (
                rearrange(residue_index, '... i -> ... i () () ()') <
                rearrange(residue_index, '... j -> ... () j () ()'))
        
    # Backbone C--N bond between subsequent residues is no clash.
    n_idx = residue_constants.atom_order['N']
    c_idx = residue_constants.atom_order['C']
    c_atom = F.one_hot(
            torch.as_tensor(c_idx, dtype=torch.long, device=pred_points.device),
            points_mask.shape[-1])
    n_atom = F.one_hot(
            torch.as_tensor(n_idx, dtype=torch.long, device=pred_points.device),
            points_mask.shape[-1])
    neighbour_mask = (
            rearrange(residue_index, '... i -> ... i () () ()') + 1 ==
            rearrange(residue_index, '... j -> ... () j () ()'))
    c_n_bonds = (
            neighbour_mask *
            rearrange(c_atom, 'm -> () () m ()') *
            rearrange(n_atom, 'n -> () () () n'))
    dist_mask *= (~c_n_bonds.bool())

    # Disulfide bridge between two cysteines is no clash.
    cys_sg_idx = residue_constants.restype_name_to_atom14_names['CYS'].index('SG')
    if cys_sg_idx < points_mask.shape[-1]:
        cys_sg_atom = F.one_hot(
                torch.as_tensor(cys_sg_idx, dtype=torch.long, device=pred_points.device),
                points_mask.shape[-1])
        disulfide_bonds = (
                rearrange(cys_sg_atom, 'm -> () () m ()') *
                rearrange(cys_sg_atom, 'n -> () () () n'))
        dist_mask *= (~disulfide_bonds.bool())

    # Compute the lower bound for the allowed distances.
    dist_lower_bound = (
            rearrange(atom_radius, '... i m -> ... i () m ()') +
            rearrange(atom_radius, '... j n -> ... () j () n'))

    # Compute the error.
    dist_errors = dist_mask * F.relu(dist_lower_bound - tau - dists)

    # Compute the mean loss.
    #clash_loss = masked_mean(mask=dist_mask, value=dist_errors, epsilon=epsilon)
    clash_loss = torch.sum(dist_errors) / (epsilon + torch.sum(dist_mask))

    if loss_only:
        return dict(between_residue_clash_loss=clash_loss)

    # Compute the per atom loss sum.
    per_atom_clash = (torch.sum(dist_errors, dim=(-4, -2)) +
            torch.sum(dist_errors, dim=(-3, -1)))

    # Compute the hard clash mask.
    clash_mask = dist_mask * (dists < dist_lower_bound - tau)
    per_atom_clash_mask = torch.maximum(torch.amax(clash_mask, dim=(-4, -2)),
            torch.amax(clash_mask, dim=(-3, -1)))

    return dict(between_residue_clash_loss=clash_loss,
            per_atom_clash=per_atom_clash,
            per_atom_clash_mask=per_atom_clash_mask)

def within_residue_clash_loss(pred_points, points_mask, residue_index, aatypes, tau1=1.5, tau2=15, epsilon=1e-6, loss_only=False):
    """Loss to penalize steric clashes within residues"""
    assert pred_points.shape[-1] == 3
    assert pred_points.shape[:-1] == points_mask.shape
    assert points_mask.shape[:-1] == residue_index.shape
    assert aatypes.shape == residue_index.shape

    # Compute the mask for each residue.
    dist_mask = (rearrange(points_mask, '... m -> ... m ()') *
            rearrange(points_mask, '... n -> ... () n'))
    dist_mask *= (~torch.eye(points_mask.shape[-1], dtype=torch.bool, device=points_mask.device))

    # Distance matrix
    dists = torch.sqrt(epsilon + torch.sum(
            (rearrange(pred_points,
                    '... m d -> ... m () d') - 
             rearrange(pred_points,
                    '... n d -> ... () n d'))**2, dim=-1))

    # Compute the loss.
    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
            overlap_tolerance=tau1,
            bond_length_tolerance_factor=tau2)
    num_atoms = points_mask.shape[-1]
    atom_lower_bound = batched_gather(
            restype_atom14_bounds['lower_bound'][...,:num_atoms,:num_atoms], aatypes)
    atom_upper_bound = batched_gather(
            restype_atom14_bounds['upper_bound'][...,:num_atoms,:num_atoms], aatypes)

    lower_errors = F.relu(atom_lower_bound - dists)
    upper_errors = F.relu(dists - atom_upper_bound)

    #clash_loss = masked_mean(mask=dist_mask,
    #        value=lower_errors+upper_errors,
    #        epsilon=epsilon)
    dist_errors = dist_mask * (lower_errors + upper_errors)
    clash_loss = torch.sum(dist_errors) / (epsilon + torch.sum(dist_mask))

    if loss_only:
        return dict(within_residue_clash_loss=clash_loss)

    # Compute the per atom loss sum.
    per_atom_clash = (torch.sum(dist_errors, dim=-2) + torch.sum(dist_errors, dim=-1))

    # Compute the violations mask.
    per_atom_clash_mask = dist_mask * ((dists < atom_lower_bound) |
                              (dists > atom_upper_bound))
    per_atom_clash_mask = torch.maximum(
            torch.amax(per_atom_clash_mask, dim=-2), torch.amax(per_atom_clash_mask, dim=-1))
    
    return dict(within_residue_clash_loss=clash_loss,
            per_atom_clash=per_atom_clash,
            per_atom_clash_mask=per_atom_clash_mask)

def symmetric_ground_truth_create_alt(seq, coord, coord_mask):
    coord_exists = batched_gather(residue_constants.restype_atom14_mask,
            seq)
    if not exists(coord):
        return dict(coord_exists=coord_exists)

    # pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14)
    renaming_transform = batched_gather(
            residue_constants.RENAMING_MATRICES, seq)

    coord_alt = torch.einsum('... m d,... m n->... n d',
            coord, renaming_transform)
    coord_alt_mask = torch.einsum('... m,... m n->... n',
            coord_mask.float(), renaming_transform)

    is_symmetric_mask = 1.0 - np.eye(residue_constants.RENAMING_MATRICES.shape[-1])
    coord_is_symmetric = batched_gather(
            np.sum(is_symmetric_mask*residue_constants.RENAMING_MATRICES, axis=-1) > 0,
            seq)

    return dict(coord_alt=coord_alt,
            coord_alt_mask=coord_alt_mask.bool(),
            coord_is_symmetric=coord_is_symmetric,
            coord_exists=coord_exists)

def symmetric_ground_truth_find_optimal(coord_pred, coord_exists,
        coord, coord_mask, coord_alt, coord_alt_mask, coord_is_symmetric, epsilon=1e-10):
    """Find optimal renaming for ground truth that maximizes LDDT. """
    assert coord_pred.shape == coord.shape
    assert coord_pred.shape == coord_alt.shape
    assert coord_exists.shape == coord_mask.shape
    assert coord_exists.shape == coord_alt_mask.shape
    assert coord_exists.shape == coord_is_symmetric.shape

    def to_distance(point):
        return torch.sqrt(epsilon + torch.sum(
                (rearrange(point, '... i c r-> ... i () c () r') - rearrange(point, '... j d r -> ... () j () d r'))**2,
                dim=-1))
    # Create the pred distance matrix.
    # shape (N, N, 14, 14)
    pred_dist = to_distance(coord_pred)
    # Compute distances for ground truth with original and alternative names.
    # shape (N, N, 14, 14)
    gt_dist = to_distance(coord)
    gt_alt_dist = to_distance(coord_alt)

    def to_lddt(x, y):
        return torch.sqrt(epsilon + (x - y)**2)
    # Compute LDDT's.
    # shape (N, N, 14, 14)
    lddt = to_lddt(pred_dist, gt_dist)
    lddt_alt = to_lddt(pred_dist, gt_alt_dist)

    # Create a mask for ambiguous atoms in rows vs. non-ambiguous atoms
    # in cols.
    # shape (N ,N, 14, 14)
    mask = (rearrange(coord_mask*coord_is_symmetric, '... i c -> ... i () c ()') * # rows
            rearrange(coord_mask*(~coord_is_symmetric), '... j d -> ... () j () d')) # cols

    # Aggregate distances for each residue to the non-amibuguous atoms.
    # shape (N)
    per_res_lddt = torch.sum(lddt * mask, dim=(-3, -2, -1))
    per_res_lddt_alt = torch.sum(lddt_alt * mask, dim=(-3, -2, -1))

    # Decide for each residue, whether alternative naming is better.
    # shape (N)
    return per_res_lddt_alt < per_res_lddt  # alt_naming_is_better

def symmetric_ground_truth_rename(coord_pred, coord_exists,
        coord, coord_mask, coord_alt, coord_alt_mask, coord_is_symmetric, epsilon=1e-10):
    """Find optimal renaming of ground truth based on the predicted positions. """
    alt_naming_is_better = symmetric_ground_truth_find_optimal(
            coord_pred, coord_exists,
            coord, coord_mask, coord_alt, coord_alt_mask, coord_is_symmetric,
            epsilon=epsilon)
    coord_renamed = (rearrange(~alt_naming_is_better, '... i -> ... i () ()')*coord +
            rearrange(alt_naming_is_better, '... i -> ... i () ()')*coord_alt)
    coord_renamed_mask = (rearrange(~alt_naming_is_better, '... i -> ... i () ()')*coord_mask +
            rearrange(alt_naming_is_better, '... i -> ... i () ()')*coord_alt_mask)

    return dict(alt_naming_is_better=alt_naming_is_better,
            coord_renamed=coord_renamed,
            coord_renamed_mask=coord_renamed_mask)
