# model.py
# Reproduced from DeepMind Grid Cells implementation
import torch
import torch.nn as nn
import numpy as np

class GridCellsRNN(nn.Module):
    def __init__(self, target_ensembles, nh_lstm, nh_bottleneck, 
                 dropout_rates=[0.5],           # model_dropout_rates
                 bottleneck_has_bias=False,     # model_bottleneck_has_bias
                 weight_decay=1e-5,             # model_weight_decay
                 init_weight_disp=0.0):         # model_init_weight_disp
        super(GridCellsRNN, self).__init__()
        self.nh_lstm = nh_lstm
        self.nh_bottleneck = nh_bottleneck
        self.dropout_rates = dropout_rates
        self.bottleneck_has_bias = bottleneck_has_bias
        self.weight_decay = weight_decay
        self.target_ensembles = target_ensembles
        
        # Input size: Speed, sin(phi), cos(phi) -> 3 dims
        self.input_size = 3 
        
        # 1. LSTM Core
        # Original: snt.LSTM(self._nh_lstm)
        # Note: Sonnet/TF LSTMs initialize with uniform or glorot. PyTorch defaults are similar.
        self.lstm = nn.LSTM(self.input_size, nh_lstm, batch_first=True)
        
        # 2. Bottleneck Layer
        # Original: snt.Linear(..., regularizers={"w": l2_regularizer})
        self.bottleneck = nn.Linear(nh_lstm, nh_bottleneck, bias=bottleneck_has_bias)
        
        # 3. Output Heads (Place Cells & Head Direction Cells)
        # Original: snt.Linear(..., initializers={"w": displaced_linear_initializer})
        self.heads = nn.ModuleList()
        for ens in target_ensembles:
            linear = nn.Linear(nh_bottleneck, ens.n_cells)
            self._init_displaced(linear, nh_bottleneck, init_weight_disp)
            self.heads.append(linear)
            
        # 4. Initialization Projections (Mapping init conditions to LSTM state)
        # Original: snt.Linear(self._nh_lstm, name="state_init")
        total_init_size = sum([ens.n_cells for ens in target_ensembles])
        self.init_lstm_hidden = nn.Linear(total_init_size, nh_lstm)
        self.init_lstm_cell = nn.Linear(total_init_size, nh_lstm)

    def _init_displaced(self, layer, input_size, displace):
        """
        Replicates: displaced_linear_initializer
        tf.truncated_normal_initializer(mean=displace*stddev, stddev=stddev)
        Standard truncation in TF is usually 2 stddevs.
        """
        stddev = 1.0 / np.sqrt(input_size)
        mean = displace * stddev
        # PyTorch trunc_normal_: values outside [a, b] are redrawn
        nn.init.trunc_normal_(layer.weight, mean=mean, std=stddev, 
                              a=mean - 2*stddev, b=mean + 2*stddev)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, concat_init, vels):
        # 1. Initialize LSTM State
        # TF: tf.squeeze(ens.get_init(...)) used in inputs, then passed to Linear
        h0 = self.init_lstm_hidden(concat_init).unsqueeze(0) # [1, B, H]
        c0 = self.init_lstm_cell(concat_init).unsqueeze(0)   # [1, B, H]
        
        # 2. Run LSTM
        # TF: tf.nn.dynamic_rnn
        lstm_out, _ = self.lstm(vels, (h0, c0))
        
        # 3. Bottleneck
        bottleneck = self.bottleneck(lstm_out)
        
        # 4. Split Dropout (Replicates TF split logic)
        # TF: scale_pops = tf.split(bottleneck, n_scales, axis=1)
        if self.training and self.dropout_rates:
            n_scales = len(self.dropout_rates)
            if n_scales > 1:
                chunk_size = self.nh_bottleneck // n_scales
                chunks = torch.split(bottleneck, chunk_size, dim=-1)
                dropped_chunks = []
                for chunk, rate in zip(chunks, self.dropout_rates):
                    dropped_chunks.append(torch.nn.functional.dropout(chunk, p=rate, training=True))
                bottleneck_dropped = torch.cat(dropped_chunks, dim=-1)
            else:
                bottleneck_dropped = torch.nn.functional.dropout(
                    bottleneck, p=self.dropout_rates[0], training=True)
        else:
            bottleneck_dropped = bottleneck
        
        # 5. Heads
        outputs = [head(bottleneck_dropped) for head in self.heads]
            
        return outputs, bottleneck, lstm_out

    def get_param_groups(self):
        """
        Strict reproduction of L2 regularization scope.
        TF Original: Only 'bottleneck' and 'pc_logits' (heads) had l2_regularizer attached.
        LSTM weights and Biases generally did NOT have L2 in the original code snippet.
        """
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            # Apply L2 only to weights of bottleneck and heads
            if ('bottleneck.weight' in name) or ('heads' in name and 'weight' in name):
                decay_params.append(param)
            else:
                no_decay_params.append(param)
                
        return [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]