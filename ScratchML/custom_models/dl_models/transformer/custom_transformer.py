import math
import torch
import copy

class Custom_SelfAttentionMH(torch.nn.Module):
    def __init__(self, d_model, n_heads, droput=0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, 'Model size (d_model) should be divisible by the number of heads (n_heads).'
        self.n_heads = n_heads
        self.query_transform = torch.nn.Linear(d_model, d_model)
        self.key_transform =  torch.nn.Linear(d_model, d_model)
        self.value_transform =  torch.nn.Linear(d_model, d_model)

        self.relevancy_activation = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(droput)
        self.out_transform = torch.nn.Linear(d_model, d_model)

        self.last_attention_map = None
    
    def forward(self, query_features, key_features, dependency_mask=None, key_padding_mask=None):        

        queries = self.query_transform(query_features)
        keys = self.key_transform(key_features)
        values = self.value_transform(key_features)

        batch_size, keys_max_len, d_model = key_features.shape
        queries_max_len = query_features.shape[1]

        if dependency_mask is None:
            dependency_mask = torch.zeros(queries_max_len, keys_max_len).to(queries.device)

        # Divide queries, keys, and values projections per each head
        # (batch_size, max_seq_len, d_model) -> (batch_size, max_seq_len, n_heads, d_model/n_heads)
        queries_per_head =  queries.view(batch_size, queries_max_len, self.n_heads, -1)
        keys_per_head = keys.view(batch_size, keys_max_len, self.n_heads, -1)
        values_per_head = values.view(batch_size, keys_max_len, self.n_heads, -1)        

        
        qk_relevancy = torch.einsum("bqhe,bkhe->bhqk", (queries_per_head, keys_per_head))
        qk_relevancy = qk_relevancy / math.sqrt(d_model)

        # Expand and apply masks
        key_padding_mask_expanded = key_padding_mask[:, None, None, :].expand_as(qk_relevancy)
        qk_relevancy.masked_fill_(key_padding_mask_expanded, float('-inf'))
        qk_relevancy = qk_relevancy + dependency_mask

        # Activate relevancy logits for an attention scores 
        relevancy_scores = self.relevancy_activation(qk_relevancy)
        relevancy_scores = self.dropout(relevancy_scores)
        self.last_attention_map = relevancy_scores.detach()
        
        # Apply attention scores for tunable values
        aggregated_query = torch.einsum("bhqk,bkhe->bqhe", (relevancy_scores, values_per_head))
        aggregated_query = aggregated_query.reshape(batch_size, queries_max_len, -1)
        res = self.out_transform(aggregated_query)
          
        return res

class CustomEncoder(torch.nn.Module):
    def __init__(self, embedding_size, n_heads, dropout):
        super().__init__()
        self.attention_module = Custom_SelfAttentionMH(embedding_size, n_heads, droput=dropout)
        self.first_dropout = torch.nn.Dropout(dropout)
        self.first_normalization = torch.nn.LayerNorm(embedding_size)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, embedding_size*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embedding_size*2, embedding_size)
        )
        self.second_dropout = torch.nn.Dropout(dropout)
        self.second_normalization = torch.nn.LayerNorm(embedding_size)
    
    def forward(self, encoder_input, dependency_mask, padding_mask):
        attention_output = self.attention_module(encoder_input, encoder_input, dependency_mask, padding_mask)
        attention_output = self.first_dropout(attention_output)
        output = self.first_normalization(attention_output + encoder_input)
        
        ffn_output = self.ffn(output)
        ffn_output = self.second_dropout(ffn_output)
        output = self.second_normalization(ffn_output + output)
        
        return output

class CustomDecoder(torch.nn.Module):
    def __init__(self, embedding_size, n_heads, dropout):
        super().__init__()
        self.attention_module = Custom_SelfAttentionMH(embedding_size, n_heads, droput=dropout)
        self.first_dropout = torch.nn.Dropout(dropout)
        self.first_normalization = torch.nn.LayerNorm(embedding_size)

        self.cross_attention_module = Custom_SelfAttentionMH(embedding_size, n_heads, droput=dropout)
        self.second_dropout = torch.nn.Dropout(dropout)
        self.second_normalization = torch.nn.LayerNorm(embedding_size)

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, embedding_size*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(embedding_size*2, embedding_size),
        )
        self.third_dropout = torch.nn.Dropout(dropout)
        self.third_normalization = torch.nn.LayerNorm(embedding_size)
    
    def forward(self, decoder_input, encoder_output, dependency_mask, src_padding_mask, tgt_padding_mask):
        self_attention_output = self.attention_module(decoder_input, decoder_input, dependency_mask, tgt_padding_mask)
        self_attention_output = self.first_dropout(self_attention_output)
        self_attention_output = self.first_normalization(self_attention_output + decoder_input)

        cross_attention_output = self.cross_attention_module(self_attention_output, encoder_output, key_padding_mask=src_padding_mask)
        cross_attention_output = self.second_dropout(cross_attention_output)
        cross_attention_output = self.second_normalization(cross_attention_output + self_attention_output)

        ffn_output = self.ffn(cross_attention_output)
        ffn_output = self.third_dropout(ffn_output)
        output = self.third_normalization(ffn_output + cross_attention_output)
        
        return output

class CustomEncoderStack(torch.nn.Module):
    def __init__(self, encoder, n_layers) -> None:
        super().__init__()
        self.module_layers = torch.nn.ModuleList([copy.deepcopy(encoder) for layer in range(n_layers)])
        self.n_layers = n_layers
        self.init_weights()
    
    def forward(self, encoder_input, dependency_mask, src_padding_mask):
        for layer in self.module_layers:
            encoder_input = layer(encoder_input, dependency_mask, src_padding_mask)
        return encoder_input
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

class CustomDecoderStack(torch.nn.Module):
    def __init__(self, decoder, n_layers) -> None:
        super().__init__()
        self.module_layers = torch.nn.ModuleList([copy.deepcopy(decoder) for layer in range(n_layers)])
        self.n_layers = n_layers
        self.init_weights()
    
    def forward(self, tgt_input, encoder_output, dependency_mask, src_padding_mask, tgt_padding_mask):
        for layer in self.module_layers:
            decoder_input = layer(tgt_input, encoder_output, dependency_mask, src_padding_mask, tgt_padding_mask)
        return decoder_input
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

class CustomTransformer(torch.nn.Module):
    def __init__(self, d_model=512, n_encoder_layers=6, n_decoder_layers=6, n_heads=8, dropout=0.1) -> None:
        super().__init__()
        encoder = CustomEncoder(d_model, n_heads, dropout=dropout)
        decoder = CustomDecoder(d_model, n_heads, dropout=dropout)
        self.encoder = CustomEncoderStack(encoder, n_layers=n_encoder_layers)
        self.decoder = CustomDecoderStack(decoder, n_layers=n_decoder_layers)
    
    def forward(self, src_embs, tgt_embs, src_mask, tgt_mask, memory_mask, src_padding_mask, tgt_key_padding_mask, memory_key_padding_mask):
        encoder_output = self.encoder(src_embs, src_mask, src_padding_mask)
        decoder_output = self.decoder(tgt_embs, encoder_output, tgt_mask, memory_key_padding_mask, tgt_key_padding_mask)
        return decoder_output