from helper import *
CUDA = torch.cuda.is_available()

class SparseAttnFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, mat_size, E, out_features):
        """
        Args:
            edge: shape=(2, total)
            edge_w: shape=(total, 1)
        """
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([mat_size[0], mat_size[1], out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None
class SparseAttn(torch.nn.Module):
    def forward(self, edge, edge_w, mat_size, E, out_features):
        return SparseAttnFunction.apply(edge, edge_w, mat_size, E, out_features)

class Cross_Att(torch.nn.Module):
    def __init__(self, embed_dim, embed_out_dim, dropout, alpha, num_key, num_query, concat=True):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(
            size=(embed_out_dim, 2 * embed_dim),device='cpu'))#device='cuda'
        torch.nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a_2 = torch.nn.Parameter(torch.zeros(size=(1, embed_out_dim),device='cpu'))#device='cuda'
        torch.nn.init.xavier_normal_(self.a_2.data, gain=1.414)
        self.trans = torch.nn.Parameter(torch.zeros(size=(embed_out_dim, embed_dim ),device='cpu'))#device='cuda'
        torch.nn.init.xavier_normal_(self.trans.data, gain=1.414)
        self.spareattn = SparseAttn()
        self.num_key = num_key
        self.num_query = num_query
        self.concat = concat
        self.dropout = torch.nn.Dropout(dropout)
        self.embed_out_dim = embed_out_dim
        self.leakyrelu = torch.nn.LeakyReLU(alpha)

    def forward(self, key_list, key_embed, query_list, query_embed):
        """
        ent_embed : shape=(num_tripes, embed_dim)
        ent_list : shape=(num_tripes, )
        rel_embed : shape=(num_tripes, embed_dim)
        rel_list : shape=(num_tripes, )
        """
        #edge_h.shape=(embed_dim * 2, num_tripes)

        edge_associated_set = self.trans.mm(key_embed.t())
        # edge_associated_set = key_embed.t()

        edge_h = torch.cat((key_embed, query_embed), dim=1).t()
        #edge_m.shape=(hidden, num_tripes)
        edge_m = self.a.mm(edge_h)


        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
        edge_e = torch.exp(powers).unsqueeze(1)
        assert not torch.isnan(edge_e).any()

        edge = torch.cat([query_list.unsqueeze(0), key_list.unsqueeze(0)], dim=0)

        e_rowsum = self.spareattn(
            edge, edge_e, (self.num_query, self.num_key), edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_associated_set).t()


        # edge_w: E * D
        h_prime = self.spareattn(
            edge, edge_w, (self.num_query, self.num_key), edge_w.shape[0], self.embed_out_dim)

        # h_prime = self.spareattn(
        #     edge, edge_w, (self.num_query, self.num_key), edge_w.shape[0], edge_w.shape[1])

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out

        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime
