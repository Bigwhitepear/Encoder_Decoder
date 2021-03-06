from helper import *
from model.cagnn import *

class BaseModel(torch.nn.Module):
    def __init__(self, param):
        super(BaseModel, self).__init__()

        self.p = param
        self.act =torch.tanh
        self.bceloss = torch.nn.BCELoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

class Cagnn_encoder(BaseModel):
    def __init__(self, edge_index, edge_type, num_rel, param=None):
        super(Cagnn_encoder, self).__init__(param)

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.p.gat_dim = self.p.embed_dim if self.p.gat_layer == 1 else self.p.gat_dim
        self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        self.device = self.edge_index.device

        #这里先不使用base_rel
        self.init_rel = get_param((num_rel * 2, self.p.init_dim))

        self.head_rel_att =[Cross_Att(self.p.init_dim,
                                      self.p.gat_dim,
                                      self.p.gat_drop,
                                      self.p.gat_alpha,
                                      self.p.num_ent,
                                      num_rel * 2,
                                      concat=True)
                            for _ in range(self.p.gat_layer)]

        self.tail_rel_att = [Cross_Att(self.p.init_dim,
                                       self.p.gat_dim,
                                       self.p.gat_drop,
                                       self.p.gat_alpha,
                                       self.p.num_ent,
                                       num_rel * 2,
                                       concat=True)
                             for _ in range(self.p.gat_layer)]

        #updata entity
        self.head_ent_att = [Cross_Att(self.p.init_dim,
                                       self.p.gat_dim,
                                       self.p.gat_drop,
                                       self.p.gat_alpha,
                                       num_rel * 2,
                                       self.p.num_ent,
                                       concat=True)
                             for _ in range(self.p.gat_layer)]
        self.tail_ent_att = [Cross_Att(self.p.init_dim,
                                       self.p.gat_dim,
                                       self.p.gat_drop,
                                       self.p.gat_alpha,
                                       num_rel * 2,
                                       self.p.num_ent,
                                       concat=True)
                             for _ in range(self.p.gat_layer)]
        self.w_rel = torch.nn.Parameter(torch.zeros(size=(self.p.init_dim + self.p.gat_layer * self.p.embed_dim,
                                                    self.p.gat_layer * self.p.embed_dim)))
        torch.nn.init.xavier_uniform_(self.w_rel.data, gain=1.414)
        self.W_rel = torch.nn.Parameter(torch.zeros(size=(self.p.init_dim, self.p.gat_layer * self.p.embed_dim)))
        torch.nn.init.xavier_uniform_(self.W_rel.data, gain=1.414)
        self.w_ent = torch.nn.Parameter(torch.zeros(size=(self.p.init_dim + self.p.gat_layer * self.p.embed_dim,
                                                    self.p.gat_layer * self.p.embed_dim)))
        torch.nn.init.xavier_uniform_(self.w_ent.data, gain=1.414)
        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

    def update_relation(self, edge_index, edge_type):
        head_ent_embed = self.init_embed[edge_index[0, :]]
        tail_ent_embed = self.init_embed[edge_index[1, :]]
        rel_embed = self.init_rel[edge_type]

        head_rel = torch.cat([att_head(edge_index[0, :], head_ent_embed, edge_type, rel_embed)
                              for att_head in self.head_rel_att], dim=1)

        tail_rel = torch.cat([att_tail(edge_index[1, :], tail_ent_embed, edge_type, rel_embed)
                              for att_tail in self.tail_rel_att], dim=1)

        rel_rep = head_rel + tail_rel
        rel_final = torch.cat((rel_rep, self.init_rel), dim=1)
        rel_final = rel_final.mm(self.w_rel)
        return rel_final

    def update_entity(self, edge_index, edge_type):
        """
        1.
        head_entity  <- rel
        tail_entity  <- rel
        2.
        head_entity  <- rel  <- tail_entity
        """
        head_ent_embed = self.init_embed[edge_index[0, :]]
        tail_ent_embed = self.init_embed[edge_index[1, :]]
        rel_embed = self.init_rel[edge_type]

        head_ent = torch.cat([att_head(edge_type, rel_embed, edge_index[0, :], head_ent_embed)
                              for att_head in self.head_ent_att], dim=1)
        tail_ent = torch.cat([att_tail(edge_type, rel_embed, edge_index[1, :], tail_ent_embed)
                              for att_tail in self.tail_ent_att], dim=1)
        ent_rep = head_ent + tail_ent

        ent_final = torch.cat((ent_rep, self.init_embed), dim=1)
        ent_final = ent_final.mm(self.w_ent)
        return ent_final

    def forword_base(self, sub, rel, drop1, drop2):
        r_emb = self.update_relation(self.edge_index, self.edge_type)
        e_emb = self.update_entity(self.edge_index, self.edge_type)

        sub_emb = torch.index_select(e_emb, 0, sub)
        rel_emb = torch.index_select(r_emb, 0, rel)

        return sub_emb, rel_emb, e_emb

class Cagnn_AcrE(Cagnn_encoder):
    def __init__(self, edge_index, edge_type, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forword_base(sub, rel, self.hidden_drop, self.feature_drop)
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score

