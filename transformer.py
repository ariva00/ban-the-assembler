import torch

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads:int, dropout:float=0.0, bias:bool=True):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_out = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.scale = self.head_dim ** -0.5

    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, attn_mask=None, attn_prev=None):
        query = query.reshape(query.size(0), query.size(1), self.num_heads, self.head_dim).transpose(1,2)
        key = key.reshape(key.size(0), key.size(1), self.num_heads, self.head_dim).transpose(1,2)
        value = value.reshape(value.size(0), value.size(1), self.num_heads, self.head_dim).transpose(1,2)

        attn = torch.matmul(query, key.transpose(2, 3))
        attn = attn * self.scale

        attn = self.dropout(attn)

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn = attn.masked_fill(attn_mask == 0, -1e9) #TODO: reconsider this, currently we are masking by putting a constant, this might not be the best choice

        pre_softmax_attn = attn
        attn = attn.softmax(dim=-1)
        attn = attn + attn_prev if attn_prev is not None else attn
        output = torch.matmul(attn, value)
        output = output.transpose(1, 2).reshape(output.size(0), output.size(2), self.embed_dim)
        output = self.linear_out(output)

        hiddens = {
            "q" : query,
            "k" : key,
            "v" : value,
            "attn" : attn,
            "pre_softmax_attn" : pre_softmax_attn
        }
        return output, hiddens

class AttentionLayer(torch.nn.Module):
    """Single dot-product attention layer (supplementary Fig.1): Q/K/V projections, scaled
    dot-product attention, residual + Norm, then a ff_mult-expansion feedforward block + residual + Norm."""
    def __init__(self, embed_dim, num_heads, dropout:float=0.0, attn_bias:bool=True, ff_mult:int=4):
        super(AttentionLayer, self).__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout, attn_bias)
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.to_q = torch.nn.Linear(embed_dim, embed_dim)
        self.to_k = torch.nn.Linear(embed_dim, embed_dim)
        self.to_v = torch.nn.Linear(embed_dim, embed_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim*ff_mult),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim*ff_mult, embed_dim)
        )

    def forward(self, x, y, attn_mask=None, x_mask=None, y_mask=None, attn_prev=None):
        if x_mask is not None:
            if y_mask is None:
                y_mask = x_mask
            input_mask = (x_mask.float().unsqueeze(-1)).bmm(y_mask.float().unsqueeze(-1).transpose(-1,-2)).long().unsqueeze(1)
            if attn_mask is None:
                attn_mask = input_mask
            else:
                attn_mask = attn_mask & input_mask
        output1, hiddens = self.attn(self.to_q(x), self.to_k(y), self.to_v(y), attn_mask=attn_mask, attn_prev=attn_prev)
        output1 = self.norm1(x + output1)
        output2 = self.feed_forward(output1)
        output2 = self.norm2(output1 + output2)
        hiddens["out"] = output2
        return output2, hiddens

class Transformer(torch.nn.Module):
    """Stack of num_layers AttentionLayers. When residual=True, accumulates the post-softmax
    attention of layer i-1 into layer i's attention weights (Eq.2: Pi_i = softmax(rho_i) + Pi_{i-1})."""
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.0, residual=False):
        super(Transformer, self).__init__()
        self.layers = torch.nn.ModuleList([AttentionLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.residual = residual

    def forward(self, x, y, attn_mask=None, x_mask=None, y_mask=None, return_hiddens=False):
        attn_hiddens = []
        for layer in self.layers:
            if self.residual and len(attn_hiddens) > 0:
                attn_prev = attn_hiddens[-1]['attn']
            else:
                attn_prev = None
            x, hiddens = layer(x, y, attn_mask=attn_mask, x_mask=x_mask, y_mask=y_mask, attn_prev=attn_prev)
            attn_hiddens.append(hiddens)
        if return_hiddens:
            return x, attn_hiddens
        else:
            return x
        
class TransformerEncoderDecoder(torch.nn.Module):
    """Stack of num_layers AttentionLayers. When residual=True, accumulates the post-softmax
    attention of layer i-1 into layer i's attention weights (Eq.2: Pi_i = softmax(rho_i) + Pi_{i-1})."""
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.0):
        super(TransformerEncoderDecoder, self).__init__()
        self.self_layers = torch.nn.ModuleList([AttentionLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.cross_layers = torch.nn.ModuleList([AttentionLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x, y, attn_mask=None, x_mask=None, y_mask=None, return_hiddens=False):
        attn_hiddens = []
        for self_layer, cross_layer in zip(self.self_layers, self.cross_layers):
            x, hiddens = self_layer(x, x, attn_mask=attn_mask, x_mask=x_mask, y_mask=x_mask)
            attn_hiddens.append(hiddens)
            x, hiddens = cross_layer(x, y, attn_mask=attn_mask, x_mask=x_mask, y_mask=y_mask)
            attn_hiddens.append(hiddens)
        if return_hiddens:
            return x, attn_hiddens
        else:
            return x

if __name__ == "__main__":
    # x, y are (batch, seq, embed_dim); cross-attention supports different sequence lengths.
    model = TransformerEncoderDecoder(embed_dim=512, num_heads=8, num_layers=6)
    x = torch.randn(2, 10, 512)
    y = torch.randn(2, 7, 512)
    output = model(x, y)
    print(output.shape)
