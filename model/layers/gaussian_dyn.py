import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from diff_gaussian_rasterization_feature import GaussianRasterizationSettings as GaussianRasterizationSettings_feature
from diff_gaussian_rasterization_feature import GaussianRasterizer as GaussianRasterizer_feature
from dataset.typing import *


def convert_pose(C2W):
    # flip_yz = torch.eye(4, device=C2W.device)
    # flip_yz[1, 1] = -1
    # flip_yz[2, 2] = -1
    # C2W = torch.matmul(C2W, flip_yz)
    # TODO: As we always use the opencv corrdinate system, we need to flip the y and z axis
    return C2W


def getProjectionMatrixK(K, H, W, znear, zfar, device="cuda"):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]

    P = torch.zeros((4, 4), device=device)

    z_sign = 1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear)
    P[3, 2] = z_sign

    return P


def get_cam_info_gaussian_v2(c2w, K, H, W, znear, zfar):
    c2w = convert_pose(c2w)
    world_view_transform = torch.inverse(c2w.float())

    world_view_transform = world_view_transform.transpose(0, 1).cuda().float()
    projection_matrix = (
        getProjectionMatrixK(K, H, W, znear, zfar)
        .transpose(0, 1)
        .cuda()
    ).float()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0).float()
    camera_center = world_view_transform.inverse()[3, :3]

    return world_view_transform, full_proj_transform, camera_center



C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


class GaussianRenderer_dyn:
    '''
    compared to GaussianRenderer_V2:
    We add render of dynamic score
    '''
    def __init__(
        self, 
        resolution: list = [512, 512],
        znear: float = 0.1,
        zfar: float = 100.0, 
        renderer_type: str = "vanilla", # only support "vanilla"
        **kwargs,
    ):  
        self.renderer_type = renderer_type

        self.resolution = resolution
        self.znear = znear
        self.zfar = zfar
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")


        self.setup_functions()


    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def render(
        self, 
        gaussians: Float[Tensor, "B N F"], 
        c2w: Float[Tensor, "B V 4 4"],
        fovx: Float[Tensor, "B V"] = None,
        fovy: Float[Tensor, "B V"] = None,
        rays_o: Float[Tensor, "B V H W 3"] = None,
        rays_d: Float[Tensor, "B V H W 3"] = None,
        bg_color: Float[Tensor, "... 3"] = None, 
        scale_modifier: float = 1.,
        K=None,
        H=None,
        W=None,
        semantics=None,
    ):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        # at least one of fovx and fovy is not none
        assert fovx is not None or fovy is not None
        if fovx is None:
            fovx = fovy
        if fovy is None:
            fovy = fovx

        device = gaussians.device
        V = c2w.shape[0]

        # Raster resolution follows the per-view H, W when provided (Waymo/wide
        # inference pass tensors of shape [V]); otherwise fall back to the
        # configured resolution. All views must share the same raster size.
        if H is not None and W is not None:
            view_h = [int(H[v]) for v in range(V)]
            view_w = [int(W[v]) for v in range(V)]
            assert len(set(view_h)) == 1 and len(set(view_w)) == 1, (
                f"All views must share the same raster size, got H={view_h}, W={view_w}."
            )
            out_h, out_w = view_h[0], view_w[0]
        else:
            out_h, out_w = int(self.resolution[0]), int(self.resolution[1])

        # loop of loop...
        images = []
        dyns = []
        depths = []


        means3D = gaussians[:, 0:3].contiguous().float()
        rgbs = gaussians[:, 3:6].contiguous().float() # [N, 3]
        opacity = gaussians[:, 6:7].contiguous().float()
        rotations = gaussians[:, 7:11].contiguous().float()
        scales = gaussians[:, 11:].contiguous().float()
        means2D = torch.zeros_like(means3D, dtype=means3D.dtype, device=device)
        semantic_feature = semantics[:, :].contiguous().float()

        for v in range(V):
            fovx_ = fovx[v].clone()
            fovy_ = fovy[v].clone()
            c2w_ = c2w[v].clone()
            K_ = K[v].clone()
            H_ = H[v] if H is not None else out_h
            W_ = W[v] if W is not None else out_w
            w2c, proj, cam_p = get_cam_info_gaussian_v2(
                c2w=c2w_, K=K_, H=H_, W=W_, znear=self.znear, zfar=self.zfar
            )
            # render novel views
            tan_half_fovx = torch.tan(fovx_ * 0.5)
            tan_half_fovy = torch.tan(fovy_ * 0.5)

            if self.renderer_type == "vanilla":
                raster_settings = GaussianRasterizationSettings_feature(
                    image_height=out_h,
                    image_width=out_w,
                    tanfovx=tan_half_fovx,
                    tanfovy=tan_half_fovy,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=w2c,
                    projmatrix=proj,
                    sh_degree=0,
                    campos=cam_p,
                    prefiltered=False,
                    debug=False,
                )
                rasterizer = GaussianRasterizer_feature(raster_settings=raster_settings)
            else:
                raise NotImplementedError

            # Rasterize visible Gaussians to image, obtain their radii (on screen).
            if self.renderer_type == "vanilla":
                rendered_image, feature_map, radii, rendered_depth = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    shs=None,
                    colors_precomp=rgbs,
                    semantic_feature = semantic_feature[:,:,None], 
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None,
                )
                rendered_normal = None
            else:
                raise NotImplementedError

            rendered_image = torch.clamp(rendered_image, min=0.0, max=1.0)
            images.append(rendered_image)
            dyns.append(feature_map)
            depths.append(rendered_depth)

        images = torch.stack(images, dim=0).view(V, 3, out_h, out_w)
        dyns = torch.stack(dyns, dim=0).view(V, 1, out_h, out_w)
        depths = torch.stack(depths, dim=0).view(V, 1, out_h, out_w)

        return {
            "image": images, # [V, 3, H, W]
            "dyn": dyns, # [V, 1, H, W]
            "depth": depths
        }


    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
    
    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float() # cpu

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

        return gaussians
  
  