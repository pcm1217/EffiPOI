import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder
from recbole.model.abstract_recommender import SequentialRecommender

class StuRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # StuRec args
        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.pq_codes = dataset.pq_codes
        self.temperature = config['temperature']
        self.index_assignment_flag = False
        self.sinkhorn_iter = config['sinkhorn_iter']
        self.fake_idx_ratio = config['fake_idx_ratio']
        

        self.train_stage = config['train_stage']
        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.decoder_poi = nn.Linear(self.hidden_size, self.n_items)

        # define layers and loss
        self.pq_code_embedding = nn.Embedding(
            self.code_dim * (1 + self.code_cap), self.hidden_size, padding_idx=0)
        self.reassigned_code_embedding = None

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.trans_matrix = nn.Parameter(torch.randn(self.code_dim, self.code_cap + 1, self.code_cap + 1))

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            raise NotImplementedError()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

            
    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        pq_code_seq = self.pq_codes[item_seq]
        pq_code_emb = self.pq_code_embedding(pq_code_seq).mean(dim=-2)
        input_emb = pq_code_emb + position_embedding
        if self.train_stage == 'transductive_ft':
            input_emb = input_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_item_emb(self):
        pq_code_emb = self.pq_code_embedding(self.pq_codes).mean(dim=-2)
        return pq_code_emb  # [B H]

    def generate_fake_neg_item_emb(self, item_index):
        rand_idx = torch.randint_like(input=item_index, high=self.code_cap)
        # flatten pq codes
        base_id = (torch.arange(self.code_dim).to(item_index.device) * (self.code_cap + 1)).unsqueeze(0)
        rand_idx = rand_idx + base_id + 1
        
        mask = torch.bernoulli(torch.full_like(item_index, self.fake_idx_ratio, dtype=torch.float))
        fake_item_idx = torch.where(mask > 0, rand_idx, item_index)
        return self.pq_code_embedding(fake_item_idx).mean(dim=-2)

    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction):
        pos_id = interaction['item_id']
        pos_pq_code = self.pq_codes[pos_id]

        pos_items_emb = self.pq_code_embedding(pos_pq_code).mean(dim=-2)
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1, keepdim=True) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1).reshape(-1, 1)

        fake_item_emb = self.generate_fake_neg_item_emb(pos_pq_code)
        fake_item_emb = F.normalize(fake_item_emb, dim=-1)
        fake_logits = (seq_output * fake_item_emb).sum(dim=1, keepdim=True) / self.temperature
        fake_logits = torch.exp(fake_logits)

        loss = -torch.log(pos_logits / (neg_logits + fake_logits)) 
        return loss.mean()
    
    
    def PredLoss(self, score_teacher, score_student):
        score_teacher = F.log_softmax(score_teacher, dim=1)
        score_student = F.log_softmax(score_student, dim=1)
        loss = F.kl_div(score_student, torch.exp(score_teacher), reduction='batchmean')
        return loss


    def mse_distillation_loss(self, student_output, teacher_output):
        mse_loss = F.mse_loss(student_output, teacher_output)
        return mse_loss

    def calculate_loss(self, interaction, teacher=None):
        #teacher.eval() 
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = F.normalize(seq_output, dim=-1)
        pos_items = interaction[self.POS_ITEM_ID]
        # Remove sequences with the same next item
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))
        contrastive_loss = self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)


        if self.loss_type == 'BPR':
            raise NotImplementedError()
        else:  # self.loss_type = 'CE'
            test_item_emb = self.calculate_item_emb()
            if self.train_stage == 'transductive_ft':
                test_item_emb = test_item_emb 
            
            if self.temperature > 0:
                
                test_item_emb = F.normalize(test_item_emb, dim=-1)
            
            student_logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            
            if self.temperature > 0:
                student_logits /= self.temperature
            
            student_loss  = self.loss_fct(student_logits, pos_items)
            #
            teacher_logits, teacher_seq_output = teacher.get_teacher_outputs(interaction)
            #teacher_loss  = self.loss_fct(teacher_logits, pos_items)
            if self.temperature > 0:
                teacher_logits /= self.temperature

            distill_loss = self.PredLoss(student_logits, teacher_logits)
            mse_loss = self.mse_distillation_loss(seq_output, teacher_seq_output)
            loss = 0.7* student_loss +  0.3 * distill_loss + 0.1*mse_loss + 0.01*contrastive_loss
            loss = student_loss

            return loss

    def predict(self, interaction):
        raise NotImplementedError()

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.calculate_item_emb()
        if self.train_stage == 'transductive_ft':
            test_items_emb = test_items_emb 
        
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_items_emb = F.normalize(test_items_emb, dim=-1)
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores