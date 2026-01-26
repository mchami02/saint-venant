import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from torch.nn.init import orthogonal_, xavier_uniform_
from torch.utils.checkpoint import checkpoint


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)

class GeGELU(nn.Module):
    """https: // paperswithcode.com / method / geglu"""
    def __init__(self):
        super().__init__()
        self.fn = nn.GELU()

    def forward(self, x):
        c = x.shape[-1]  # channel last arrangement
        return self.fn(x[..., :int(c//2)]) * x[..., int(c//2):]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim*2),
            GeGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]

class CrossLinearAttention(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,  # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',  # ['xavier', 'orthogonal']
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1 / 64,  # 1/64 is for 64 x 64 ns2d,
                 cat_pos=False,
                 pos_dim=2,
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type

        self.heads = heads
        self.dim_head = dim_head

        # query is the classification token
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        if attn_type == 'galerkin':
            self.k_norm = nn.InstanceNorm1d(dim_head)
            self.v_norm = nn.InstanceNorm1d(dim_head)
        elif attn_type == 'fourier':
            self.q_norm = nn.InstanceNorm1d(dim_head)
            self.k_norm = nn.InstanceNorm1d(dim_head)
        else:
            raise Exception(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim*heads, dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain
        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim, min_freq=min_freq, scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        for param in self.to_kv.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for k
                    init_fn(param[h*self.dim_head:(h+1)*self.dim_head, :], gain=self.init_gain)
                    param.data[h*self.dim_head:(h+1)*self.dim_head, :] += self.diagonal_weight * \
                                                                          torch.diag(torch.ones(
                                                                              param.size(-1), dtype=torch.float32))

                    # for v
                    init_fn(param[(self.heads + h) * self.dim_head:(self.heads + h + 1) * self.dim_head, :], gain=self.init_gain)
                    param.data[(self.heads + h) * self.dim_head:(self.heads + h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                           torch.diag(torch.ones(
                                                                               param.size(-1), dtype=torch.float32))
                                                                               
        for param in self.to_q.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for q
                    init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)
                    param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                torch.diag(torch.ones(
                                                                                    param.size(-1), dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x, z, x_pos=None, z_pos=None):
        # x (z^T z)
        # x [b, n1, d]
        # z [b, n2, d]
        n2 = z.shape[1]   # z [b, n2, d]

        q = self.to_q(x)

        kv = self.to_kv(z).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        if (x_pos is None or z_pos is None) and self.relative_emb:
            raise Exception('Must pass in coordinates when under relative position embedding mode')
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        if self.attn_type == 'galerkin':
            k = self.norm_wrt_domain(k, self.k_norm)
            v = self.norm_wrt_domain(v, self.v_norm)
        else:  # fourier
            q = self.norm_wrt_domain(q, self.q_norm)
            k = self.norm_wrt_domain(k, self.k_norm)

        if self.relative_emb:
            if self.relative_emb_dim == 2:

                x_freqs_x = self.emb_module.forward(x_pos[..., 0], x.device)
                x_freqs_y = self.emb_module.forward(x_pos[..., 1], x.device)
                x_freqs_x = repeat(x_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                x_freqs_y = repeat(x_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                z_freqs_x = self.emb_module.forward(z_pos[..., 0], z.device)
                z_freqs_y = self.emb_module.forward(z_pos[..., 1], z.device)
                z_freqs_x = repeat(z_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                z_freqs_y = repeat(z_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                q = apply_2d_rotary_pos_emb(q, x_freqs_x, x_freqs_y)
                k = apply_2d_rotary_pos_emb(k, z_freqs_x, z_freqs_y)

            elif self.relative_emb_dim == 1:
                assert x_pos.shape[-1] == 1 and z_pos.shape[-1] == 1
                x_freqs = self.emb_module.forward(x_pos[..., 0], x.device)
                x_freqs = repeat(x_freqs, 'b n d -> b h n d', h=q.shape[1])

                z_freqs = self.emb_module.forward(z_pos[..., 0], x.device)
                z_freqs = repeat(z_freqs, 'b n d -> b h n d', h=q.shape[1])

                q = apply_rotary_pos_emb(q, x_freqs)  # query from x domain
                k = apply_rotary_pos_emb(k, z_freqs)  # key from z domain
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')
        elif self.cat_pos:
            assert x_pos.size(-1) == self.pos_dim and z_pos.size(-1) == self.pos_dim
            x_pos = x_pos.unsqueeze(1)
            x_pos = x_pos.repeat([1, self.heads, 1, 1])
            q = torch.cat([x_pos, q], dim=-1)

            z_pos = z_pos.unsqueeze(1)
            z_pos = z_pos.repeat([1, self.heads, 1, 1])
            k = torch.cat([z_pos, k], dim=-1)
            v = torch.cat([z_pos, v], dim=-1)

        dots = torch.matmul(k.transpose(-1, -2), v)

        out = torch.matmul(q, dots) * (1./n2)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class CrossFormer(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,
                 heads,
                 dim_head,
                 mlp_dim,
                 residual=True,
                 use_ffn=True,
                 use_ln=False,
                 relative_emb=False,
                 scale=1.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 dropout=0.,
                 cat_pos=False,
                 ):
        super().__init__()

        self.cross_attn_module = CrossLinearAttention(dim,
                                                      attn_type,
                                                      heads=heads,
                                                      dim_head=dim_head,
                                                      dropout=dropout,
                                                      relative_emb=relative_emb,
                                                      scale=scale,
                                                      relative_emb_dim=relative_emb_dim,
                                                      min_freq=min_freq,
                                                      init_method='orthogonal',
                                                      cat_pos=cat_pos,
                                                      pos_dim=relative_emb_dim,
                                                      )
        self.use_ln = use_ln
        self.residual = residual
        self.use_ffn = use_ffn

        if self.use_ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

        if self.use_ffn:
            self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, z, x_pos=None, z_pos=None):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        if self.use_ln:
            z = self.ln1(z)
            if self.residual:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos)) + x
            else:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos))
        else:
            if self.residual:
                x = self.cross_attn_module(x, z, x_pos, z_pos) + x
            else:
                x = self.cross_attn_module(x, z, x_pos, z_pos)

        if self.use_ffn:
            x = self.ffn(x) + x

        return x

class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):

        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class ShallowWater2DDecoder(nn.Module):
    """
        Shallow Water 2D Decoder
    """
    # Should define a recurrent neural network propagating on the graph
    def __init__(self, net_params):
        super().__init__()

        # Generalize to a convolution network on the graph
        self.hidden_dim_enc_out = net_params['hidden_dim_enc_out']
        self.hidden_dim_dec = net_params['hidden_dim_dec']
        self.out_dim = net_params['out_dim']
        self.num_heads = net_params['num_heads']
        self.decoding_depth = net_params['decoding_depth']
        self.rolling_checkpoint = net_params['rolling_checkpoint'] if 'rolling_checkpoint' in net_params else False
        self.residual = net_params['residual']
        self.relative_emb = net_params['relative_emb']
        self.relative_emb_dim = net_params['relative_emb_dim']
        self.min_freq = net_params['min_freq']
        self.cat_pos = net_params['cat_pos']
        self.out_step = net_params['out_step']


        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.hidden_dim_dec//2, scale=1),
            nn.Linear(self.hidden_dim_dec, self.hidden_dim_dec, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_dim_dec, self.hidden_dim_dec//2, bias=False),
        )

        self.decoding_transformer = CrossFormer(dim=self.hidden_dim_dec//2,
                                                attn_type='galerkin',
                                                heads=self.num_heads,
                                                dim_head=self.hidden_dim_dec//2,
                                                mlp_dim=self.hidden_dim_dec//2,
                                                residual=self.residual,
                                                use_ffn=True,
                                                use_ln=True,
                                                relative_emb=self.relative_emb,
                                                scale=1.,
                                                relative_emb_dim=self.relative_emb_dim,
                                                min_freq=self.min_freq,
                                                dropout=0.,
                                                cat_pos=self.cat_pos,
                                                )


        self.expand_feat = nn.Linear(self.hidden_dim_dec//2, self.hidden_dim_dec)

        self.propagator = nn.ModuleList([
               nn.ModuleList([nn.LayerNorm(self.hidden_dim_dec),
               nn.Sequential(
                    nn.Linear(self.hidden_dim_dec + 2, self.hidden_dim_dec, bias=False),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim_dec, self.hidden_dim_dec, bias=False),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim_dec, self.hidden_dim_dec, bias=False))])
            for _ in range(self.decoding_depth)])

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim_dec),
            nn.Linear(self.hidden_dim_dec, self.hidden_dim_dec//2, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_dim_dec // 2, self.hidden_dim_dec // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_dim_dec//2, self.out_dim * self.out_step, bias=True))

    def propagate(self, h, propagate_pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            h = h + ffn(torch.concat((norm_fn(h), propagate_pos), dim=-1))
        return h

    def decode(self, h):
        h = self.to_out(h)
        return h

    def rollout(self, h, input_pos, propagate_pos, forward_steps):

        history = []
        h_add = self.coordinate_projection.forward(propagate_pos)
        h = self.decoding_transformer.forward(x=h_add,
                                              z=h,
                                              x_pos=propagate_pos,
                                              z_pos=input_pos)
        h = self.expand_feat(h)

        for _ in range(forward_steps//self.out_step):
            if self.rolling_checkpoint and self.training:
                h = checkpoint(self.propagate, h, propagate_pos)
                h_out = checkpoint(self.decode, h)
            else:
                h = self.propagate(h, propagate_pos)
                h_out = self.decode(h)
            h_out = rearrange(h_out, 'b n (t c) -> b n t c', c=self.out_dim, t=self.out_step)
            history.append(h_out)
        return history

def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_heads,
                 use_bias,
                 r=0.1,
                 relative_emb=False,
                 relative_emb_dim=None,
                 min_freq=None,
                 cat_pos=False,
                 scale=1.):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.r = r
        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        self.min_freq = min_freq
        self.cat_pos = cat_pos
        self.scale = scale

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        if self.relative_emb:
            assert not self.cat_pos
            self.emb_module = RotaryEmbedding(out_dim // relative_emb_dim, min_freq=min_freq, scale=scale)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, h, pos=None, g=None):

        bs, N, C = h.shape
        assert C == self.in_dim

        assert g
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        Q_h = rearrange(Q_h, 'b n (h d) -> b h n d', h=self.num_heads, d=self.out_dim)
        K_h = rearrange(K_h, 'b n (h d) -> b h n d', h=self.num_heads, d=self.out_dim)
        V_h = rearrange(V_h, 'b n (h d) -> b h n d', h=self.num_heads, d=self.out_dim)

        if self.relative_emb:
            if self.relative_emb_dim == 2:
                freqs_x = self.emb_module.forward(pos[..., 0], h.device)
                freqs_y = self.emb_module.forward(pos[..., 1], h.device)
                freqs_x = repeat(freqs_x, 'b n d -> b h n d', h=Q_h.shape[1])
                freqs_y = repeat(freqs_y, 'b n d -> b h n d', h=Q_h.shape[1])
                Q_h = apply_2d_rotary_pos_emb(Q_h, freqs_x, freqs_y)
                K_h = apply_2d_rotary_pos_emb(K_h, freqs_x, freqs_y)
            elif self.relative_emb_dim == 1:
                assert pos.shape[-1] == 1
                freqs = self.emb_module.forward(pos[..., 0], h.device)
                freqs = repeat(freqs, 'b n d -> b h n d', h=Q_h.shape[1])
                Q_h = apply_rotary_pos_emb(Q_h, freqs)
                K_h = apply_rotary_pos_emb(K_h, freqs)
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')

        elif self.cat_pos:
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.num_heads , 1, 1])
            Q_h, K_h, V_h = [torch.cat([pos, x], dim=-1) for x in (Q_h, K_h, V_h)]

        else:
            pass

        Q_h = rearrange(Q_h, 'b h n d -> n b h d')
        K_h = rearrange(K_h, 'b h n d -> n b h d')
        V_h = rearrange(V_h, 'b h n d -> n b h d')

        g.ndata['Q_h'] = Q_h  # N, bs, num_heads, out_dim (out_dim+2)
        g.ndata['K_h'] = K_h  # N, bs, num_heads, out_dim (out_dim+2)
        g.ndata['V_h'] = V_h  # N, bs, num_heads, out_dim (out_dim+2)

        self.propagate_attention(g)

        attn_out = g.ndata['wV'] / g.ndata['z']

        attn_out = rearrange(attn_out, 'n b h d -> b n (h d)')

        return attn_out



class InputEncoderLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_heads,
                 dropout=0.0,
                 use_ball_connectivity=True,
                 r=0.1,
                 use_knn=False,
                 k=101,
                 norm_type='batch',
                 residual=True,
                 use_bias=False,
                 relative_emb=False,
                 relative_emb_dim=None,
                 min_freq=None,
                 cat_pos=False,
                 scale=1):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.r = r
        self.norm_type = norm_type
        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        self.min_freq = min_freq
        self.cat_pos = cat_pos
        self.residual = residual
        self.scale = scale

        if self.norm_type == 'layer':
            self.norm1 = nn.LayerNorm(out_dim)
            self.norm2 = nn.LayerNorm(out_dim)
        elif self.norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(out_dim)
            self.norm2 = nn.BatchNorm1d(out_dim)
        elif self.norm_type == 'group':
            self.norm1 = nn.GroupNorm(4, out_dim)
            self.norm2 = nn.GroupNorm(4, out_dim)
        elif self.norm_type == 'instance':
            self.norm1 = nn.InstanceNorm1d(out_dim)
            self.norm2 = nn.InstanceNorm1d(out_dim)
        elif self.norm_type == 'no':
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            raise ValueError(f'Norm type {self.norm_type} not supported')

        self.attention = MultiHeadAttentionLayer(in_dim=in_dim,
                                                 out_dim=out_dim//num_heads,
                                                 num_heads=num_heads,
                                                 use_bias=use_bias,
                                                 r=r,
                                                 relative_emb=relative_emb,
                                                 relative_emb_dim=relative_emb_dim,
                                                 min_freq=min_freq,
                                                 cat_pos=cat_pos,
                                                 scale=scale)

        if not cat_pos:
            self.O = nn.Linear(out_dim, out_dim)
        else:
            self.O = nn.Linear(out_dim + 2 * num_heads, out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)

        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

    def forward(self, h, pos=None, g=None):
        bs, N, C = h.shape

        h_in1 = h.clone()  # for first residual connection

        # multi-head attention out
        h = self.attention(h, pos=pos, g=g)
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        h = self.O(h)

        if self.residual:
            h = h_in1 + h

        if self.norm_type in ['layer', 'no']:
            h = self.norm1(h)  # Bs, N, C
        elif self.norm_type in ['batch', 'instance']:
            h = self.norm1(h.view(bs, C, N)).view(bs, N, C)  # Bs, N, C -> Bs, C, N -> Bs, N, C
        elif self.norm_type in ['group']:
            raise NotImplementedError
        else:
            raise ValueError(f'Norm type {self.norm_type} not supported')

        h_in2 = h.clone()  # for second residual connection
        
        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h

        if self.norm_type in ['layer', 'no']:
            h = self.norm2(h)  # Bs, N, C
        elif self.norm_type in ['batch', 'instance']:
            h = self.norm2(h.view(bs, C, N)).view(bs, N, C)  # Bs, N, C -> Bs, C, N -> Bs, N, C
        elif self.norm_type in ['group']:
            raise NotImplementedError
        else:
            raise ValueError(f'Norm type {self.norm_type} not supported')

        return h.view(bs, -1, self.out_channels)
        
    def __repr__(self):
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, heads={self.num_heads}, residual={self.residual})'

class PDEEncoder(nn.Module):
    """
        Encoder for masked PDE grids using graph attention.
        Takes observed points (values + positions) and produces latent representation.
    """

    def __init__(self, 
                 in_dim=1,
                 hidden_dim=128,
                 hidden_dim_out=128,
                 num_heads=4,
                 num_layers=4,
                 in_feat_dropout=0.0,
                 dropout=0.0,
                 norm_type='layer',
                 residual=True,
                 relative_emb=True,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 cat_pos=False):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_out = hidden_dim_out
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.residual = residual
        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        self.min_freq = min_freq
        self.cat_pos = cat_pos

        # Projection for input features (values)
        self.proj_h = nn.Linear(in_features=in_dim, out_features=hidden_dim)

        # Dropout
        self.in_feat_dropout_layer = nn.Dropout(in_feat_dropout)

        # Encoder layers (using full attention, no graph structure needed)
        self.encoder_layers = nn.ModuleList([
            SelfAttentionEncoderLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                norm_type=norm_type,
                residual=residual,
                relative_emb=relative_emb,
                relative_emb_dim=relative_emb_dim,
                min_freq=min_freq,
                cat_pos=cat_pos
            )
            for _ in range(num_layers)
        ])

        self.to_out = nn.Linear(hidden_dim, hidden_dim_out, bias=False)

    def forward(self, h, pos):
        """
        Args:
            h: (B, N_obs, in_dim) - observed values
            pos: (B, N_obs, 2) - positions (t, x) of observed points
        Returns:
            h_out: (B, N_obs, hidden_dim_out) - encoded features
        """
        # Embedding
        h = self.proj_h(h)  # (B, N, D) -> (B, N, hidden_dim)

        # Dropout
        h = self.in_feat_dropout_layer(h)

        # Encoder layers
        for layer in self.encoder_layers:
            h = layer(h, pos=pos)

        h_out = self.to_out(h)
        return h_out  # (B, N, hidden_dim_out)


class SelfAttentionEncoderLayer(nn.Module):
    """
        Self-attention encoder layer without graph structure.
        Uses full attention between all observed points.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_heads,
                 dropout=0.0,
                 norm_type='layer',
                 residual=True,
                 relative_emb=True,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 cat_pos=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm_type = norm_type
        self.residual = residual
        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        self.min_freq = min_freq
        self.cat_pos = cat_pos

        # Normalization
        if norm_type == 'layer':
            self.norm1 = nn.LayerNorm(out_dim)
            self.norm2 = nn.LayerNorm(out_dim)
        elif norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(out_dim)
            self.norm2 = nn.BatchNorm1d(out_dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Multi-head self-attention (full attention)
        self.attention = FullSelfAttention(
            dim=in_dim,
            heads=num_heads,
            dim_head=out_dim // num_heads,
            dropout=dropout,
            relative_emb=relative_emb,
            relative_emb_dim=relative_emb_dim,
            min_freq=min_freq,
            cat_pos=cat_pos,
        )

        # Output projection
        if not cat_pos:
            self.O = nn.Linear(out_dim, out_dim)
        else:
            self.O = nn.Linear(out_dim + 2 * num_heads, out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

    def forward(self, h, pos=None):
        bs, N, C = h.shape

        h_in1 = h.clone()

        # Self-attention
        h = self.attention(h, pos=pos)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.O(h)

        if self.residual:
            h = h_in1 + h

        # Normalization
        if self.norm_type == 'layer':
            h = self.norm1(h)
        elif self.norm_type == 'batch':
            h = self.norm1(h.transpose(1, 2)).transpose(1, 2)

        h_in2 = h.clone()

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h

        if self.norm_type == 'layer':
            h = self.norm2(h)
        elif self.norm_type == 'batch':
            h = self.norm2(h.transpose(1, 2)).transpose(1, 2)

        return h


class FullSelfAttention(nn.Module):
    """
    Full self-attention with optional relative positional embedding.
    """
    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 relative_emb=True,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 scale=1.,
                 cat_pos=False):
        super().__init__()
        
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        self.cat_pos = cat_pos
        
        if relative_emb:
            self.emb_module = RotaryEmbedding(dim_head // relative_emb_dim, min_freq=min_freq, scale=scale)

    def forward(self, x, pos=None):
        b, n, _ = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # Apply rotary positional embedding if enabled
        if self.relative_emb and pos is not None:
            if self.relative_emb_dim == 2:
                freqs_t = self.emb_module.forward(pos[..., 0], x.device)
                freqs_x = self.emb_module.forward(pos[..., 1], x.device)
                freqs_t = repeat(freqs_t, 'b n d -> b h n d', h=self.heads)
                freqs_x = repeat(freqs_x, 'b n d -> b h n d', h=self.heads)
                q = apply_2d_rotary_pos_emb(q, freqs_t, freqs_x)
                k = apply_2d_rotary_pos_emb(k, freqs_t, freqs_x)
            elif self.relative_emb_dim == 1:
                freqs = self.emb_module.forward(pos[..., 0], x.device)
                freqs = repeat(freqs, 'b n d -> b h n d', h=self.heads)
                q = apply_rotary_pos_emb(q, freqs)
                k = apply_rotary_pos_emb(k, freqs)
        
        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class PDEDecoder(nn.Module):
    """
        Decoder for PDE prediction.
        Takes encoded observations and predicts values at target positions.
    """
    def __init__(self,
                 hidden_dim=128,
                 out_dim=1,
                 num_heads=4,
                 decoding_depth=4,
                 residual=True,
                 relative_emb=True,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 cat_pos=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.decoding_depth = decoding_depth
        self.residual = residual
        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        self.min_freq = min_freq
        self.cat_pos = cat_pos

        # Project target coordinates to feature space
        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, hidden_dim // 2, scale=1),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
        )

        # Cross-attention: target positions attend to encoded observations
        self.decoding_transformer = CrossFormer(
            dim=hidden_dim // 2,
            attn_type='galerkin',
            heads=num_heads,
            dim_head=hidden_dim // 2,
            mlp_dim=hidden_dim // 2,
            residual=residual,
            use_ffn=True,
            use_ln=True,
            relative_emb=relative_emb,
            scale=1.,
            relative_emb_dim=relative_emb_dim,
            min_freq=min_freq,
            dropout=0.,
            cat_pos=cat_pos,
        )

        # Expand features after cross-attention
        self.expand_feat = nn.Linear(hidden_dim // 2, hidden_dim)

        # Propagator layers for refinement
        self.propagator = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                nn.Sequential(
                    nn.Linear(hidden_dim + 2, hidden_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim, bias=False)
                )
            ])
            for _ in range(decoding_depth)
        ])

        # Output projection
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim, bias=True)
        )

    def forward(self, z, z_pos, target_pos):
        """
        Args:
            z: (B, N_obs, hidden_dim) - encoded observations from encoder
            z_pos: (B, N_obs, 2) - positions of observations (t, x)
            target_pos: (B, N_target, 2) - positions to predict (t, x)
        Returns:
            out: (B, N_target, out_dim) - predicted values at target positions
        """
        # Project target coordinates
        h = self.coordinate_projection(target_pos)  # (B, N_target, hidden_dim//2)

        # Cross-attention: target positions query the encoded observations
        h = self.decoding_transformer(
            x=h,
            z=z,
            x_pos=target_pos,
            z_pos=z_pos
        )  # (B, N_target, hidden_dim//2)

        # Expand features
        h = self.expand_feat(h)  # (B, N_target, hidden_dim)

        # Propagator refinement
        for norm_fn, ffn in self.propagator:
            h = h + ffn(torch.cat([norm_fn(h), target_pos], dim=-1))

        # Output projection
        out = self.to_out(h)  # (B, N_target, out_dim)
        return out


class MaskedGridPredictor(nn.Module):
    """
    Complete model for predicting full PDE grid from masked observations.
    
    Takes a masked grid where -1 indicates unknown values, extracts the known
    points with their (t, x) positions, encodes them, and decodes to predict
    the full grid.
    """
    def __init__(self,
                 nt,
                 nx,
                 dt,
                 dx,
                 in_channels=1,
                 out_channels=1,
                 hidden_dim=128,
                 num_heads=4,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dropout=0.0,
                 relative_emb=True,
                 mask_value=-1.0):
        super().__init__()
        
        self.nt = nt
        self.nx = nx
        self.dt = dt
        self.dx = dx
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_value = mask_value
        
        # Create full coordinate grid (normalized to [0, 1])
        t_coords = torch.linspace(0, 1, nt)
        x_coords = torch.linspace(0, 1, nx)
        t_grid, x_grid = torch.meshgrid(t_coords, x_coords, indexing='ij')
        # (nt, nx, 2) -> (nt*nx, 2)
        full_coords = torch.stack([t_grid, x_grid], dim=-1).reshape(-1, 2)
        self.register_buffer('full_coords', full_coords)
        
        # Encoder
        self.encoder = PDEEncoder(
            in_dim=in_channels,
            hidden_dim=hidden_dim,
            hidden_dim_out=hidden_dim // 2,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            dropout=dropout,
            relative_emb=relative_emb,
            relative_emb_dim=2,
        )
        
        # Decoder
        self.decoder = PDEDecoder(
            hidden_dim=hidden_dim,
            out_dim=out_channels,
            num_heads=num_heads,
            decoding_depth=num_decoder_layers,
            relative_emb=relative_emb,
            relative_emb_dim=2,
        )

    def extract_observed_points(self, masked_grid):
        """
        Extract observed (non-masked) points and their positions.
        
        Args:
            masked_grid: (B, C, nt, nx) - masked input grid
            
        Returns:
            obs_values: (B, N_obs, C) - observed values
            obs_pos: (B, N_obs, 2) - positions of observed points
        """
        B, C, nt, nx = masked_grid.shape
        device = masked_grid.device
        
        # Find mask for each sample (any channel == mask_value means masked)
        # We use the first channel to determine mask
        mask = (masked_grid[:, 0] != self.mask_value)  # (B, nt, nx)
        
        # Get maximum number of observed points across batch
        n_obs_per_sample = mask.sum(dim=(1, 2))  # (B,)
        max_n_obs = n_obs_per_sample.max().item()
        
        # Create output tensors
        obs_values = torch.zeros(B, max_n_obs, C, device=device)
        obs_pos = torch.zeros(B, max_n_obs, 2, device=device)
        
        # Create coordinate grids
        t_coords = torch.linspace(0, 1, nt, device=device)
        x_coords = torch.linspace(0, 1, nx, device=device)
        t_grid, x_grid = torch.meshgrid(t_coords, x_coords, indexing='ij')
        
        for b in range(B):
            sample_mask = mask[b]  # (nt, nx)
            indices = sample_mask.nonzero()  # (n_obs, 2) - indices into (nt, nx)
            n_obs = indices.shape[0]
            
            # Extract values
            for c in range(C):
                obs_values[b, :n_obs, c] = masked_grid[b, c][sample_mask]
            
            # Extract positions
            obs_pos[b, :n_obs, 0] = t_grid[sample_mask]
            obs_pos[b, :n_obs, 1] = x_grid[sample_mask]
        
        return obs_values, obs_pos

    def forward(self, masked_grid):
        """
        Forward pass: predict full grid from masked observations.
        
        Args:
            masked_grid: (B, C, nt, nx) - masked input grid where -1 indicates unknown
            
        Returns:
            pred_grid: (B, out_channels, nt, nx) - predicted full grid
        """
        B = masked_grid.shape[0]

        # Extract observed points
        obs_values, obs_pos = self.extract_observed_points(masked_grid)
        # obs_values: (B, N_obs, C)
        # obs_pos: (B, N_obs, 2)
        
        # Encode observations
        z = self.encoder(obs_values, obs_pos)  # (B, N_obs, hidden_dim//2)
        
        # Get target positions (full grid)
        target_pos = self.full_coords.unsqueeze(0).expand(B, -1, -1)  # (B, nt*nx, 2)
        
        # Decode to full grid
        pred = self.decoder(z, obs_pos, target_pos)  # (B, nt*nx, out_channels)
        
        # Reshape to grid
        pred_grid = pred.permute(0, 2, 1).reshape(B, self.out_channels, self.nt, self.nx)
        
        return pred_grid


class ShallowWater2DEncoder(nn.Module):
    """
        Shallow Water 2D Encoder (Legacy - kept for compatibility)
    """

    def __init__(self, net_params):
        super().__init__()

        # Parameter
        self.net_params = net_params

        self.in_dim = net_params['in_dim']
        self.hidden_dim_enc = net_params['hidden_dim_enc']
        self.hidden_dim_enc_out = net_params['hidden_dim_enc_out']
        self.out_dim = net_params['out_dim']
        self.num_heads = net_params['num_heads']
        self.num_layers = net_params['num_layers']
        self.in_feat_dropout = net_params['in_feat_dropout']
        self.dropout = net_params['dropout']
        self.use_ball_connectivity = net_params['use_ball_connectivity']
        self.r = net_params['r']
        self.use_knn = net_params['use_knn']
        self.k = net_params['k']
        assert self.use_ball_connectivity != self.use_knn, "use_ball_connectivity and use_knn cannot be both True"
        self.norm_type = net_params['norm_type']
        self.residual = net_params['residual']
        self.relative_emb = net_params['relative_emb']
        self.relative_emb_dim = net_params['relative_emb_dim']
        self.min_freq = net_params['min_freq']
        self.cat_pos = net_params['cat_pos']
        self.device = net_params['device']

        # Projection
        self.proj_h = nn.Linear(in_features=self.in_dim, out_features=self.hidden_dim_enc)
        self.proj_h_add = nn.Linear(in_features=2, out_features=self.hidden_dim_enc)

        # Dropout
        self.in_feat_dropout = nn.Dropout(self.in_feat_dropout)

        # Encoder
        self.input_encoder_layers = nn.ModuleList([InputEncoderLayer(in_dim=self.hidden_dim_enc,
                                                                     out_dim=self.hidden_dim_enc,
                                                                     num_heads=self.num_heads,
                                                                     dropout=self.dropout,
                                                                     use_ball_connectivity=self.use_ball_connectivity,
                                                                     r=self.r,
                                                                     use_knn=self.use_knn,
                                                                     k=self.k,
                                                                     min_freq=self.min_freq,
                                                                     relative_emb=self.relative_emb,
                                                                     relative_emb_dim=self.relative_emb_dim,
                                                                     cat_pos=self.cat_pos,
                                                                     norm_type=self.norm_type,
                                                                     residual=self.residual)
                                                   for _ in range(self.num_layers - 1)])

        self.to_out = nn.Linear(self.hidden_dim_enc, self.hidden_dim_enc_out, bias=False)

    def forward(self, h, input_pos=None, g=None):
        # Embedding
        h = self.proj_h(h)  # (B, N, D) -> (B, N, D')

        # Dropout
        h = self.in_feat_dropout(h)

        # Encoder
        for graph_transformer_layer in self.input_encoder_layers:
            h = graph_transformer_layer(h, pos=input_pos, g=g)

        h_out = self.to_out(h)

        return h_out  # (B, N, D')

