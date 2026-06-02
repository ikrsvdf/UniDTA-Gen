import numpy as np
import torch
import torch.nn as nn
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer
from typing import Optional, Dict
from rdkit import Chem
from dgl.nn.functional import edge_softmax
import dgl.function as fn
from dgl.nn import SumPooling, AvgPooling, MaxPooling
import torch.nn.functional as F
class KAN_linear(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=False):
        super(KAN_linear, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, outdim, inputdim, gridsize) /
            (np.sqrt(inputdim) * np.sqrt(gridsize))
        )
        
        init_k = torch.arange(1, gridsize + 1, dtype=torch.float32)
        self.k = nn.Parameter(init_k.view(1, 1, 1, gridsize))
        
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))
    
    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        k = torch.abs(self.k) + 1e-6
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)  
        c = torch.cos(k * xrshp)  
        s = torch.sin(k * xrshp)
        
        c = c.view(1, x.shape[0], x.shape[1], self.gridsize)
        s = s.view(1, x.shape[0], x.shape[1], self.gridsize)
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        
        if self.addbias:
            y += self.bias
        
        y = y.view(outshape)
        return y


class Gat_Kan_layer(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_node_feats, out_edge_feats, num_heads, grid_size, bias=True):
        super(Gat_Kan_layer, self).__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_node = nn.Linear(in_node_feats+in_edge_feats, out_node_feats * num_heads, bias=True)
        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats * num_heads, bias=False)
        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats * num_heads, bias=False)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats * num_heads, bias=False)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_edge_feats)))
        self.output_node = KAN_linear(out_node_feats, out_node_feats, grid_size, addbias=True)
        self.output_edge = KAN_linear(out_edge_feats, out_edge_feats, grid_size, addbias=True)
        self.edge_kan = KAN_linear(out_edge_feats * num_heads, out_edge_feats * num_heads, gridsize=1, addbias=True)
        self.node_kan = KAN_linear(in_node_feats+in_edge_feats, in_node_feats+in_edge_feats, gridsize=1, addbias=True)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc_node.weight)
        nn.init.xavier_normal_(self.fc_ni.weight)
        nn.init.xavier_normal_(self.fc_fij.weight)
        nn.init.xavier_normal_(self.fc_nj.weight)
        nn.init.xavier_normal_(self.attn)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def message_func(self, edges):
        return {'feat': edges.data['feat']}
    
    def reduce_func(self, nodes):
        num_edges = nodes.mailbox['feat'].size(1)  
        agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges  
        return {'agg_feats': agg_feats}
    
    def forward(self, graph, nfeats, efeats, get_attention=False):
        with graph.local_scope():
            graph.ndata['feat'] = nfeats
            graph.edata['feat'] = efeats
            
            in_degrees = graph.in_degrees().float().unsqueeze(-1)
            in_degrees[in_degrees == 0] = 1  
            
            f_ni = self.fc_ni(nfeats)  
            f_nj = self.fc_nj(nfeats) 
            f_fij = self.fc_fij(efeats)  
            
            graph.srcdata.update({'f_ni': f_ni})
            graph.dstdata.update({'f_nj': f_nj})
            graph.apply_edges(fn.u_add_v('f_ni', 'f_nj', 'f_tmp'))
            
            f_out = graph.edata.pop('f_tmp') + f_fij
            f_out = self.edge_kan(f_out) 
            
            if self.bias is not None:
                f_out = f_out + self.bias
            f_out = nn.functional.leaky_relu(f_out)
            f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
            
            e = (f_out * self.attn).sum(dim=-1).unsqueeze(-1)
            
            graph.send_and_recv(graph.edges(), self.message_func, reduce_func=self.reduce_func)
            m_feats = torch.cat((graph.ndata['feat'], graph.ndata['agg_feats']), dim=1)
            m_feats = self.node_kan(m_feats)  
            
            graph.edata['a'] = edge_softmax(graph, e)
            graph.ndata['h_out'] = self.fc_node(m_feats).view(-1, self._num_heads, self._out_node_feats)
            graph.update_all(fn.u_mul_e('h_out', 'a', 'm'),
                           fn.sum('m', 'h_out'))
            
            h_out = nn.functional.leaky_relu(graph.ndata['h_out'])
            h_out = h_out.view(-1, self._num_heads, self._out_node_feats)
            
            h_out = torch.sum(h_out, dim=1)
            f_out = torch.sum(f_out, dim=1)
            
            out_n = self.output_node(h_out)
            out_e = self.output_edge(f_out)
            
            if get_attention:
                return out_n, out_e, graph.edata.pop('a')
            else:
                return out_n, out_e


class KA_GAT(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, hidden_dim, out_1, out_2, gride_size, head, layer_num, pooling):
        super(KA_GAT, self).__init__()
        self.in_node_dim = in_node_dim
        self.in_edge_dim = in_edge_dim
        self.hidden_dim = hidden_dim
        self.out_1 = out_1
        self.out_2 = out_2
        self.head = head
        self.layer = layer_num
        self.grid_size = gride_size
        self.pooling = pooling
        
        self.node_kan_line = KAN_linear(in_node_dim, hidden_dim, gride_size, addbias=False)
        self.edge_kan_line = KAN_linear(in_edge_dim, hidden_dim, gride_size, addbias=False)
        
        self.attentions = nn.ModuleList()
        
        self.attentions.append(Gat_Kan_layer(in_node_feats=in_node_dim, in_edge_feats=in_edge_dim,
                                            out_node_feats=hidden_dim, out_edge_feats=hidden_dim,
                                            num_heads=self.head, grid_size=self.grid_size))
        
        for _ in range(self.layer-1):
            self.attentions.append(Gat_Kan_layer(in_node_feats=hidden_dim, in_edge_feats=hidden_dim,
                                                out_node_feats=hidden_dim, out_edge_feats=hidden_dim,
                                                num_heads=self.head, grid_size=self.grid_size))
        
        self.leaky_relu = nn.LeakyReLU()
        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
    
    def forward(self, g, node_feature, edge_feature):
        for i in range(len(self.attentions)):
            atten = self.attentions[i]
            node_feature, edge_feature = atten(g, node_feature, edge_feature)
        
        out1 = F.leaky_relu(node_feature)
        
        if self.pooling == 'avg':
            y = self.avgpool(g, out1)  
        elif self.pooling == 'max':
            y = self.maxpool(g, out1)
        elif self.pooling == 'sum':
            y = self.sumpool(g, out1)
        else:
            print('No pooling found!!!!')
            y = out1
        
        return y, out1



def _get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class PositionalEncodingBatchFirst(nn.Module):
    def __init__(self, d_hid, n_position=1025):
        super().__init__()
        pos_table = _get_sinusoid_encoding_table(n_position, d_hid)
        self.register_buffer('pos_table', pos_table)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pos_table[:, :seq_len]
class PositionalEncodingSeqFirst(nn.Module):
    def __init__(self, d_hid, n_position=1025):
        super().__init__()
        pos_table = _get_sinusoid_encoding_table(n_position, d_hid)
        # print("pos_table shape",pos_table.shape)
        self.register_buffer('pos_table', pos_table)
        self.register_buffer('pe', pos_table.transpose(0,1))
    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pos_table[:, :seq_len].transpose(0, 1)

    
    
class Namespace:
    def __init__(self, argvs):
        for k, v in argvs.items():
            setattr(self, k, v)
            

class TransformerEncoder(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()

        self.layer = nn.ModuleList([
            TransformerEncoderLayer(Namespace({
                'encoder_embed_dim': dim,
                'encoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'encoder_normalize_before': True,
                'encoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, encoder_padding_mask=None):
        for layer in self.layer:
            x = layer(x, encoder_padding_mask)
        x = self.layer_norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()

        self.layer = nn.ModuleList([
            TransformerDecoderLayer(Namespace({
                'decoder_embed_dim': dim,
                'decoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'decoder_normalize_before': True,
                'decoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, mem, x_mask=None, x_padding_mask=None, mem_padding_mask=None):
        for layer in self.layer:
            x = layer(x, mem,
                      self_attn_mask=x_mask, self_attn_padding_mask=x_padding_mask,
                      encoder_padding_mask=mem_padding_mask)[0]
        x = self.layer_norm(x)
        return x

    @torch.jit.export
    def forward_one(self,
                    x: torch.Tensor,
                    mem: torch.Tensor,
                    incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
                    mem_padding_mask: torch.BoolTensor = None,
                    ) -> torch.Tensor:
        x = x[-1:]
        for layer in self.layer:
            x = layer(x, mem, incremental_state=incremental_state, encoder_padding_mask=mem_padding_mask)[0]
        x = self.layer_norm(x)
        return x


def format_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


class DrugGenModel(nn.Module):
    def __init__(self,tokenizer, d_model=128, d_inner=512, n_layers=8, n_head=8,
                 d_k=80, d_v=80, dropout=0.1, n_position=1025,max_drug=128,ka_gat_model=None):
        
        
        super(DrugGenModel, self).__init__()
        
        self.ka_gat_model = ka_gat_model
        self.protein_model_2 = ProteinFeatureExtractor(
            d_model=d_model, d_inner=d_inner, d_k=d_k, d_v=d_v, dropout=dropout,
            n_position=n_position
        )
        self.max_drug = max_drug
        self.pos_encoding = PositionalEncodingSeqFirst(d_model, n_position=max_drug)

        vocab_size = len(tokenizer)
        self.word_pred = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        torch.nn.init.zeros_(self.word_pred[3].bias)
        self.word_embed = nn.Embedding(vocab_size, d_model)
        self.sos_value = tokenizer.s2i['<sos>']
        self.eos_value = tokenizer.s2i['<eos>']
        self.pad_value = tokenizer.s2i['<pad>']
        self.zz_seg_encoding = nn.Parameter(torch.randn(d_model))
        self.dencoder = TransformerEncoder(dim=d_model, ff_dim=d_inner, num_head=n_head, num_layer=n_layers)
        self.decoder = Decoder(dim=d_model, ff_dim=d_inner, num_head=n_head, num_layer=n_layers)
        self.cond = nn.Linear(128, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1)
        )
        
        self.mean = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.var = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.chem_proj = nn.Sequential(
            nn.Linear(3, d_model),       
            nn.LeakyReLU(),              
            nn.Linear(d_model, d_model)  
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model), 
            nn.ReLU(),
            nn.Linear(d_model, 1),  
            nn.Sigmoid()  
        )
       
    def reparameterize(self, z_mean, logvar, con, a, qed, sas, logp, eps=1e-8):

        batch_size = z_mean.size(0)
        
        logvar = torch.clamp(logvar, -10, 10) 
        kl_element = 0.5 * (logvar.exp() + z_mean.pow(2) - 1 - logvar)
        kl_loss = torch.sum(kl_element) / batch_size
        std = torch.exp(0.5 * logvar) + eps
        epsilon = torch.randn_like(z_mean)
        z_ = z_mean + std * epsilon
        chem_embed = torch.stack([qed, sas, logp], dim=1)
        con_embedding = self.cond(con)
        chem_embed = self.chem_proj(chem_embed)
        
        gate = torch.sigmoid(self.gate_proj(torch.cat([con_embedding, chem_embed], dim=-1)))
        z_ = z_ + gate * con_embedding + (1 - gate) * chem_embed
        
        return z_, kl_loss
    
    def get_mask(self,smiles_seq):
        pad_token_id = 2
        pp_mask = (smiles_seq != pad_token_id).float()
        return pp_mask
    
    def expand_then_fusing(self, z, pp_mask, vvs):
        zz = z
        zzs = zz + self.zz_seg_encoding
        full_mask = zz.new_zeros(zz.shape[1], zz.shape[0])
        full_mask = torch.cat((pp_mask, full_mask), dim=1)  # batch seq_plus
        zzz = torch.cat((vvs, zzs), dim=0)  
        zzz = self.dencoder(zzz, full_mask)
        
        return zzz, full_mask
    
    def sample(self, batch_size, device):
        z = torch.randn(1, self.hidden_dim).to(device)
        return z

    def forward(self, smiles_seq, esm_feat, di_feat, mask, affinity, qed, sas, logp, graphs=None):
        Protein_vector = self.protein_model_2(esm_feat, di_feat, mask)
        con = Protein_vector

        batched_graph = graphs 
        

        node_feats = batched_graph.ndata['feat']
        edge_feats = batched_graph.edata['feat']

        graph_feature, out1 = self.ka_gat_model(batched_graph, node_feats, edge_feats)
        num_nodes_per_graph = batched_graph.batch_num_nodes().tolist() 
        node_features_batch = []

        start_idx = 0
        for num_nodes in num_nodes_per_graph:
            end_idx = start_idx + num_nodes
            subgraph_nodes = out1[start_idx:end_idx]  
            node_features_batch.append(subgraph_nodes)
            start_idx = end_idx
        

        node_features_batch = torch.stack(node_features_batch, dim=0)  

        node_features_3d = node_features_batch.permute(1, 0, 2) 
        
        mu = self.mean(node_features_3d)
        logvar = self.var(node_features_3d)
        
        feature_drug, kl_loss = self.reparameterize(mu, logvar, con, affinity, qed, sas, logp) 
        
        mask = self.get_mask(smiles_seq)
        
        zzz, encoder_mask = self.expand_then_fusing(feature_drug, mask, feature_drug)
        
        _, target_length = smiles_seq.shape
        target_mask = torch.triu(torch.ones(target_length, target_length, dtype=torch.bool), diagonal=1).to(smiles_seq.device)
        target_embed = self.word_embed(smiles_seq)
        target_embed = self.pos_encoding(target_embed.permute(1, 0, 2).contiguous())
        
        output = self.decoder(target_embed, zzz, x_mask=target_mask, mem_padding_mask=encoder_mask).permute(1, 0, 2).contiguous()
        
        prediction_scores = self.word_pred(output)
        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        
        targets = smiles_seq[:, 1:].contiguous()
        batch_size, sequence_length, vocab_size = shifted_prediction_scores.size()
        shifted_prediction_scores = shifted_prediction_scores.view(-1, vocab_size)
        targets = targets.view(-1)
        
        lm_loss = F.cross_entropy(shifted_prediction_scores, targets, ignore_index=self.pad_value)
   
        v_graph = graph_feature  
        
        combined = torch.cat([
            v_graph,  
            Protein_vector,  
        ], dim=1)  
        
        output = self.mlp(combined)
        
        return output.squeeze(), prediction_scores, lm_loss, kl_loss
    

    def _generate(self, zzz, encoder_mask, random_sample=True, return_score=False,
                    temperature=0.7, top_k=20, top_p=0.99, max_retry=10,
                    apply_chemical_constraints=True,
                    chem_check_interval=5):
        
        def _init_generation(batch_size, device):
            token = torch.full((batch_size, self.max_drug), self.pad_value, 
                            dtype=torch.long, device=device)
            token[:, 0] = self.sos_value
            text_embed = self.word_embed(token[:, 0])
            text_pos = self.pos_encoding.pe
            text_embed = text_embed + text_pos[0]
            text_embed = text_embed.unsqueeze(0)
            incremental_state = torch.jit.annotate(
                Dict[str, Dict[str, Optional[torch.Tensor]]],
                torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {}),
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            return token, text_embed, incremental_state, finished
        
        def _sample_next_token(logits, temperature, top_k, top_p, random_sample):
            logits = logits / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                indices_to_remove = logits < v[:, [-1]]
                logits[indices_to_remove] = -float('Inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            if random_sample:
                k = torch.multinomial(probs, 1).squeeze(1)
            else:
                k = torch.argmax(logits, -1)
            return k
        
        def _apply_constraints(batch_idx, t, logits, token, grammar_states, chem_checker):
            if not apply_chemical_constraints or chem_checker is None:
                return logits
            
            if t < 3 or t % chem_check_interval != 0:
                return logits
            
            current_sequence = token[batch_idx, :t]
            current_smiles = self._partial_tokens_to_smiles(current_sequence)
            
            grammar_states[batch_idx] = self._update_grammar_state(
                grammar_states[batch_idx], 
                token[batch_idx, t-1] if t > 1 else None
            )
            
            if not self._is_grammar_ready_for_chemical_check(grammar_states[batch_idx]):
                return logits
            
            for vocab_idx in range(logits.shape[1]):
                if vocab_idx in [self.pad_value, self.eos_value, self.sos_value]:
                    continue
                
                new_token_str = self.tokenizer.i2s.get(vocab_idx, '')
                
                if not self._is_token_grammatically_valid(
                    grammar_states[batch_idx], new_token_str, current_smiles):
                    logits[batch_idx, vocab_idx] = logits[batch_idx, vocab_idx] - 50.0
                    continue
                
                test_sequence = current_sequence.clone()
                test_sequence = torch.cat([
                    test_sequence, 
                    torch.tensor([vocab_idx], device=device)
                ])
                
                if not self._is_token_chemically_valid(
                    test_sequence, current_smiles, vocab_idx, chem_checker):
                    penalty = -2.0 * (t / self.max_drug)
                    logits[batch_idx, vocab_idx] = logits[batch_idx, vocab_idx] + penalty
            
            return logits
        
        batch_size = zzz.shape[1]
        device = zzz.device
        
        for attempt in range(max_retry):
            try:
                token, text_embed, incremental_state, finished = _init_generation(batch_size, device)
                
                chem_checker = None
                grammar_states = None
                
                if apply_chemical_constraints and hasattr(self, 'tokenizer'):
                    chem_checker = self._init_chemical_checker()
                    grammar_states = [self._init_grammar_state() for _ in range(batch_size)]
                
                for t in range(1, self.max_drug):
                    one = self.decoder.forward_one(text_embed, zzz, incremental_state, 
                                                mem_padding_mask=encoder_mask)
                    one = one.squeeze(0)
                    logits = self.word_pred(one)
                    
                    if apply_chemical_constraints and chem_checker is not None:
                        for batch_idx in range(batch_size):
                            if not finished[batch_idx]:
                                logits = _apply_constraints(
                                    batch_idx, t, logits, token, grammar_states, chem_checker
                                )
                    
                    k = _sample_next_token(logits, temperature, top_k, top_p, random_sample)
                    
                    token[:, t] = k
                    finished |= k == self.eos_value
                    
                    if finished.all():
                        break
                    
                    text_embed = self.word_embed(k)
                    text_embed = text_embed + self.pos_encoding.pe[t]
                    text_embed = text_embed.unsqueeze(0)
                    
                    if apply_chemical_constraints and grammar_states is not None:
                        for batch_idx in range(batch_size):
                            if not finished[batch_idx]:
                                grammar_states[batch_idx] = self._update_grammar_state(
                                    grammar_states[batch_idx], k[batch_idx]
                                )
                
                return token[:, 1:], True
                
            except Exception as e:
                if attempt == max_retry - 1:
                    empty_token = torch.full((batch_size, self.max_drug-1), self.pad_value, 
                                            dtype=torch.long, device=device)
                    return empty_token, False
        
        empty_token = torch.full((batch_size, self.max_drug-1), self.pad_value, 
                                dtype=torch.long, device=device)
        return empty_token, False


    def _init_grammar_state(self):
        return {
            'open_paren': 0,      
            'open_branch': 0,      
            'ring_starts': {},      
            'ring_ends': {},        
            'chiral_center': False,  
            'bond_type': None,    
            'in_brackets': False,   
            'atom_count': 0,        
            'last_atom': None,    
        }


    def _update_grammar_state(self, state, last_token):
        if last_token is None or not hasattr(self, 'tokenizer'):
            return state
        
        token_str = self.tokenizer.i2s.get(last_token.item(), '')
        
        if token_str == '(':
            state['open_paren'] += 1
            state['open_branch'] += 1
        elif token_str == ')':
            if state['open_paren'] > 0:
                state['open_paren'] -= 1
            if state['open_branch'] > 0:
                state['open_branch'] -= 1
        elif token_str == '[':
            state['in_brackets'] = True
        elif token_str == ']':
            state['in_brackets'] = False
        elif token_str in ['-', '=', '#', ':', '/', '\\']:
            state['bond_type'] = token_str
        elif token_str == '@':
            state['chiral_center'] = True
        elif token_str == '@@':
            state['chiral_center'] = True
    
        if (token_str and 
            not token_str in ['(', ')', '[', ']', '-', '=', '#', ':', '/', '\\', '@', '@@'] and
            not token_str.isdigit() and
            token_str not in ['<sos>', '<eos>', '<pad>']):
            
            state['atom_count'] += 1
            state['last_atom'] = token_str
            state['bond_type'] = None
            state['chiral_center'] = False
        if token_str.isdigit():
            ring_num = int(token_str)
            if ring_num not in state['ring_starts']:
                state['ring_starts'][ring_num] = state['atom_count']
            else:
                state['ring_ends'][ring_num] = state['atom_count']
        
        return state


    def _is_grammar_ready_for_chemical_check(self, grammar_state):
        if (grammar_state['open_paren'] > 0 or 
            grammar_state['open_branch'] > 0 or
            grammar_state['in_brackets']):
            return False
        
        if grammar_state['atom_count'] < 2:
            return False
        
        return True


    def _is_token_grammatically_valid(self, grammar_state, new_token, current_smiles):
        if not new_token:
            return False
        if new_token in ['<eos>', '<pad>']:
            return True
        if new_token == ')':
            return grammar_state['open_paren'] > 0
        elif new_token == ']':
            return grammar_state['in_brackets']
        elif new_token in ['-', '=', '#', ':', '/', '\\']:
            return grammar_state['last_atom'] is not None
        elif new_token in ['@', '@@']:
            return grammar_state['last_atom'] is not None
        elif new_token == '(':
            return grammar_state['last_atom'] is not None
        elif new_token == '[':
            return not grammar_state['in_brackets']
        elif new_token.isdigit():
            ring_num = int(new_token)
            if ring_num in grammar_state['ring_starts']:
                return True
            else:
                return grammar_state['last_atom'] is not None
        else:
            return True


    def _init_chemical_checker(self):
        class EnhancedChemicalChecker:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.forbidden_smarts = [
                    '[O]-[O]',
                    '[N]-[O,Br,Cl,I,F,P]',
                    '[S,P]-[Br,Cl,I,F]',
                    '[P]-[O]-[P]',
                    '[Br,Cl,I,F]-[Br,Cl,I,F]',
                    '[O]-[O,Br,Cl,I,F]',
                    '[N,P]-[Br,Cl,I,F,P]',
                    '[C]-[S]=[C]',
                    '[C,N,O,S]=[C]=[C,N,O,S]',
                    '[C,N,S,I]=[P]',
                    '[I]=[N,C,O]',
                    '[C]-[I]-[C]'
                ]
                
                self.max_valence = {
                    'C': 4, 'N': 5, 'O': 2, 'F': 1, 
                    'P': 5, 'S': 6, 'Cl': 1, 'Br': 1, 'I': 7
                }

                self.triple_bond_cache = {}
            
            def check_partial_smiles(self, smiles: str, new_token: str) -> bool:
                
                if not smiles or len(smiles) < 2:
                    return True
                test_smiles = smiles + new_token

                try:
                    mol = Chem.MolFromSmiles(test_smiles, sanitize=False)
                    if mol is None:
                        return self._is_possibly_valid_incomplete_smiles(test_smiles)
                    try:
                        Chem.SanitizeMol(mol, Chem.SANITIZE_SYMMRINGS | 
                                        Chem.SANITIZE_SETAROMATICITY | 
                                        Chem.SANITIZE_SETCONJUGATION | 
                                        Chem.SANITIZE_SETHYBRIDIZATION)
                    except:
                        pass
                    if not self._check_valence(mol):
                        return False
                    if not self._check_triple_bond_constraints(mol):
                        return False
                    if self._contains_forbidden_patterns(mol):
                        return False
                    
                    return True
                    
                except:
                    return self._is_possibly_valid_incomplete_smiles(test_smiles)
            
            def _is_possibly_valid_incomplete_smiles(self, smiles: str) -> bool:
                open_paren = smiles.count('(')
                close_paren = smiles.count(')')
                open_bracket = smiles.count('[')
                close_bracket = smiles.count(']')
                if abs(open_paren - close_paren) > 2 or abs(open_bracket - close_bracket) > 2:
                    return False
                ring_digits = []
                for char in smiles:
                    if char.isdigit():
                        ring_digits.append(char)
                
                from collections import Counter
                digit_counts = Counter(ring_digits)
                for digit, count in digit_counts.items():
                    if count > 2:  
                        return False
                
                return True
            
            def _check_valence(self, mol) -> bool:
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    if symbol in self.max_valence:
                        valence = atom.GetTotalValence()
                        if valence > self.max_valence[symbol] + 1:
                            return False
                return True
            
            def _check_triple_bond_constraints(self, mol) -> bool:
                from rdkit.Chem import BondType

                triple_bonds = []
                for bond in mol.GetBonds():
                    if bond.GetBondType() == BondType.TRIPLE:
                        atom1 = bond.GetBeginAtom()
                        atom2 = bond.GetEndAtom()
                        triple_bonds.append((atom1, atom2))
                for atom1, atom2 in triple_bonds:

                    double_bond_atoms = set()
                    
                    for atom in [atom1, atom2]:
                        for neighbor in atom.GetNeighbors():
                            if neighbor not in [atom1, atom2]:
                                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                                if bond and bond.GetBondType() == BondType.DOUBLE:
                                    double_bond_atoms.add(neighbor.GetIdx())

                    if len(double_bond_atoms) >= 2:
                        return False
                
                return True
            
            def _contains_forbidden_patterns(self, mol) -> bool:
                from rdkit import Chem
                
                for smarts in self.forbidden_smarts:
                    try:
                        patt = Chem.MolFromSmarts(smarts)
                        if patt and mol.HasSubstructMatch(patt):
                            return True
                    except:
                        continue
                return False
        
        return EnhancedChemicalChecker(self.tokenizer)


    def _partial_tokens_to_smiles(self, tokens):
        if not hasattr(self, 'tokenizer'):
            return ""
        
        try:
            valid_tokens = []
            for token in tokens:
                token_val = token.item()
                if token_val == self.sos_value or token_val == self.pad_value:
                    continue
                if token_val == self.eos_value:
                    break
                token_str = self.tokenizer.i2s.get(token_val, '')
                valid_tokens.append(token_str)
            
            return ''.join(valid_tokens)
        except:
            return ""


    def _is_token_chemically_valid(self, token_sequence, current_smiles, new_token_idx, chem_checker):
        if not hasattr(self, 'tokenizer'):
            return True
        
        try:
            new_token_str = self.tokenizer.i2s.get(new_token_idx, '')
            if new_token_str in ['<eos>', '<pad>']:
                return True
            return chem_checker.check_partial_smiles(current_smiles, new_token_str)
        except:
            return True


    def generate(self, smiles_seq, esm_feat, di_feat, mask, affinity, qed, sas, logp, 
                random_sample=True, max_retry=10, require_valid=True, graphs=None,
                apply_chemical_constraints=True, 
                chem_check_interval=3,
                temperature=0.7,
                top_k=50,
                top_p=0.99):
        
        Protein_vector = self.protein_model_2(esm_feat, di_feat, mask)
        con = Protein_vector
        batched_graph = graphs 
        
        node_feats = batched_graph.ndata['feat']
        edge_feats = batched_graph.edata['feat']

        graph_feature, out1 = self.ka_gat_model(batched_graph, node_feats, edge_feats)

        num_nodes_per_graph = batched_graph.batch_num_nodes().tolist()  
        node_features_batch = []

        start_idx = 0
        for num_nodes in num_nodes_per_graph:
            end_idx = start_idx + num_nodes
            subgraph_nodes = out1[start_idx:end_idx]  
            node_features_batch.append(subgraph_nodes)
            start_idx = end_idx
        
        node_features_batch = torch.stack(node_features_batch, dim=0) 
        node_features_3d = node_features_batch.permute(1, 0, 2)  
        
        mu = self.mean(node_features_3d)
        logvar = self.var(node_features_3d)
        
        AMVO, kl_loss = self.reparameterize(mu, logvar, con, affinity, qed, sas, logp)  
        
        mask = self.get_mask(smiles_seq)
        
        zzz, encoder_mask = self.expand_then_fusing(AMVO, mask, AMVO)
        
        MAX_CONSTRAINED_ATTEMPTS = 10  
        
        constrained_success = False
        constrained_predict = None
        constrained_attempts = 0
        
        if apply_chemical_constraints:
            for attempt in range(MAX_CONSTRAINED_ATTEMPTS):
                current_temp = temperature + 0.1 * (attempt % 3)
                predict, success = self._generate(
                    zzz, encoder_mask, 
                    random_sample=random_sample, 
                    temperature=current_temp,  
                    top_k=top_k,
                    top_p=top_p,
                    max_retry=10,
                    apply_chemical_constraints=apply_chemical_constraints, 
                    chem_check_interval=chem_check_interval
                )
                
                constrained_attempts = attempt + 1
                
                if not success and not require_valid:
                    return predict, constrained_attempts
                
                if hasattr(self, 'tokenizer'):
                    generated_text = self.tokenizer.get_text(predict)
                    if require_valid and generated_text:
                        valid_smiles = []
                        for smi in generated_text:
                            formatted = format_smiles(smi)
                            if formatted:  
                                valid_smiles.append(formatted)
                        
                        if valid_smiles:
                            constrained_success = True
                            constrained_predict = predict
                            break
        
        if constrained_success:
            print(f"Constrained generation succeeded after {constrained_attempts} attempts.")
            return constrained_predict, constrained_attempts
        
        
        for attempt in range(max_retry):
            current_temp = temperature + 0.1 * (attempt % 3)
            predict, success = self._generate(
                zzz, encoder_mask, 
                random_sample=random_sample, 
                temperature=current_temp,  
                top_k=top_k,
                top_p=top_p,
                max_retry=10,
                apply_chemical_constraints=False,  
                chem_check_interval=chem_check_interval
            )
            
            total_attempts = constrained_attempts + (attempt + 1)
            
            if not success and not require_valid:
                return predict, total_attempts
            
            if hasattr(self, 'tokenizer'):
                generated_text = self.tokenizer.get_text(predict)
                if require_valid and generated_text:
                    valid_smiles = []
                    for smi in generated_text:
                        formatted = format_smiles(smi)
                        if formatted:  
                            valid_smiles.append(formatted)
                    
                    if valid_smiles:
                        return predict, total_attempts
            
            elif not require_valid:
                return predict, total_attempts
        
    
        return predict, max_retry + constrained_attempts
def create_mask_from_tensor(sequences):
    mask = (sequences.sum(dim=-1) != 0).float() 
    return mask




class LayerNormNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, drop_out=0.1):
        super(LayerNormNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(F.relu(self.ln1(self.fc1(x))))
        x = self.dropout(F.relu(self.ln2(self.fc2(x))))
        x = self.fc3(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1)  
            if attn.dtype == torch.float16:
                fill_value = torch.tensor(-1e4, dtype=torch.float16, device=attn.device)
            else:
                fill_value = -1e9
            attn = attn.masked_fill(mask == 0, fill_value)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.d_k).transpose(1, 2)  # (bs, n_head, len_q, d_k)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.d_k).transpose(1, 2)  # (bs, n_head, len_k, d_k)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head, self.d_v).transpose(1, 2)  # (bs, n_head, len_v, d_v)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # (bs, len_q, n_head * d_v)

        q = self.dropout(self.fc(q))  
        q += residual
        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.cross_attn1 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn1 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.cross_attn2 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn2 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, seq_input, struc_input, slf_attn_mask=None):
        struc_input, enc_slf_attn = self.cross_attn1(
            seq_input, struc_input, struc_input, mask=slf_attn_mask)
        struc_output = self.pos_ffn1(struc_input)

        seq_output, enc_slf_attn = self.cross_attn2(
            struc_output, seq_input, seq_input, mask=slf_attn_mask)
        seq_output = self.pos_ffn2(seq_output)
        return seq_output, struc_output, enc_slf_attn






class PorEncoder(nn.Module):

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner,
                 dropout=0.1, n_position=1025, scale_emb=False):
        super().__init__()

        self.position_enc1 = PositionalEncodingBatchFirst(d_model, n_position=n_position)
        self.position_enc2 = PositionalEncodingBatchFirst(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, seq_input, struc_input, src_mask, return_attns=False):
        enc_slf_attn_list = []
        # print("struc_input in encoder", struc_input)
        seq_input = self.dropout(self.position_enc1(seq_input))
        seq_input = self.layer_norm1(seq_input)
        struc_input = self.dropout(self.position_enc2(struc_input))
        struc_input = self.layer_norm2(struc_input)
        for i, enc_layer in enumerate(self.layer_stack):
            
            seq_input, struc_input, enc_slf_attn = enc_layer(
                seq_input, struc_input, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return seq_input, struc_input, enc_slf_attn_list
        return seq_input, struc_input


class ProteinFeatureExtractor(nn.Module):
    def __init__(self, d_model=128, d_inner=512, n_layers=1, n_head=4,
                 d_k=80, d_v=80, dropout=0.1, n_position=1025):
        super().__init__()
        self.sequence_linear = LayerNormNet(1280, 512, d_model)
        self.structure_linear = LayerNormNet(1024, 512, d_model)
        self.encoder = PorEncoder(
            n_position=n_position, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq_features, struc_features, mask=None):
        # print("struc_features", struc_features)

        src_seq = self.sequence_linear(seq_features)  # [batch, L, d_model]
        src_struc = self.structure_linear(struc_features)  # [batch, L, d_model]


        if mask is None:
            print("No mask provided, generating default mask assuming all positions are valid.")
            mask = torch.ones(seq_features.size(0), seq_features.size(1)).to(seq_features.device)

        enc_output, local_output, *_ = self.encoder(src_seq, src_struc, mask.unsqueeze(-2))

        fuse_feature = enc_output + local_output
        fuse_f = self.mlp1(fuse_feature) + fuse_feature

        masked_enc_output = fuse_f * mask.unsqueeze(-1)
        protein_features = masked_enc_output.sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)
        return protein_features





