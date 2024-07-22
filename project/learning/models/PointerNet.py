import torch
import torch.nn as nn
from torch.nn import Parameter

# PyTorch implementation of Pointer-Net based on https://arxiv.org/pdf/1506.03134.pdf and https://github.com/shirgur/PointerNet/tree/master

class Encoder(nn.Module):
    """
    Encoder of pointer net.
    LSTM taking as input a tensor of shape [B, Li, C]
    """
    def __init__(self,
                 embedding_dim:int=8,
                 hidden_dim:int=32,
                 n_layers:int=1,
                 dropout:float=0.,
                 bidir:bool=False) -> None:
        """
        embedding_dim (C): Number of embbeding channels
        hidden_dim (H): Number of hidden units for the LSTM
        n_layers (N): Number of layers for LSTMs
        dropout: Float between 0-1
        bidir: Bidirectional
        """
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim if not(bidir) else hidden_dim // 2, # To have consistent output shape
                            n_layers,
                            dropout=dropout if n_layers > 1 else 0.,
                            bidirectional=bidir,
                            batch_first=True)
        

    def forward(self,
                embedded_inputs,
                hidden=None
                ):
        """
        Inputs:
            - embedded_inputs: Embedded inputs of Pointer-Net [B, Li, C]
            - hidden: Initiated hidden units for the LSTMs [H, C]
        
        Return:
            - LSTMs outputs [B, Li, H]
            - hidden units [N, H]
        """
        if hidden == None:
            outputs, hidden = self.lstm(embedded_inputs)
        else:
            outputs, hidden = self.lstm(embedded_inputs, hidden)
        return outputs, hidden


class Attention(nn.Module):
    """
    Additive attention mechanism used in the decoder to 
    select elements from the input sequence in the decoder.
    """

    def __init__(self,
                 input_dim_encoder,
                 input_dim_decoder,
                 hidden_dim) -> None:
        """
        input_dim_encoder : Input encoder dimention
        input_dim_decoder : Input decoder dimention
        hidden_dim : Number of hidden units in the attention
        """
        super(Attention, self).__init__()

        self.W1 = nn.Linear(input_dim_encoder, hidden_dim, bias=False)
        self.W2 = nn.Linear(input_dim_decoder, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 1, bias=False)

        self.tanh = nn.Tanh()

    def forward(self, encoder_hiddens, hidden_decoder):
        """
        Inputs:
            - encoder_hiddens: all encoder hidden states [B, Li, H]
            - hidden_decoder: decoder hidden state i [B, 1, H]
        Return:
            - logits of the attention distribution over the input seqence [B, Li]
        """
        elementwise_product = self.V(self.tanh(self.W1(encoder_hiddens) + self.W2(hidden_decoder))) # [B, Li, H]
        logits = torch.sum(elementwise_product, dim=-1) # [B, Li]
        return logits
    
class Decoder(nn.Module):
    """
    Decoder of pointer net.
    LSTM taking successively as input a tensor of shape [B, 1, C]
    from the input sequence.
    """
    def __init__(self,
                 embedding_dim:int,
                 hidden_dim:int,
                 n_layers:int,
                 output_length:int) -> None:
        """
        embedding_dim (C): Number of embeddings in Pointer-Net
        hidden_dim (H): Number of hidden units for the decoder's RNN
        n_layers (N): Number of layers for LSTMs
        output_length (Lo): Length of the ouput sequence
        """
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.Lo = output_length

        self.att = Attention(hidden_dim, self.n_layers * hidden_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
                    self.embedding_dim,
                    self.hidden_dim,
                    self.n_layers,
                    dropout=0.,
                    bidirectional=False,
                    batch_first=True)

    def forward(self,
                embedded_inputs,
                decoder_input,
                hidden,
                encoder_hiddens):
        """
        Inputs:
            - embedded_inputs: Embedded inputs of Pointer-Net [B, Li, C]
            - decoder_input: First decoder's input [1, B, C]
            - hidden: First decoder's hidden states [N, B, H]
            - encoder_hiddens: Encoder's outputs [B, Li, H]
        Return:
            - (output logits, pointers indices), last hidden state
        """

        B = embedded_inputs.size(0)
        Li = embedded_inputs.size(1)

        logits = torch.empty((B, Li, self.Lo), dtype=torch.float)
        pointers = torch.empty((B, self.Lo), dtype=torch.long)

        # Recurrence loop
        for step in range(self.Lo):
            _, hidden = self.lstm(decoder_input, hidden) # LSTM on input sequence of length 1
            decoder_hidden = hidden[0].view(B, 1, -1)
            logits_att = self.att(encoder_hiddens, decoder_hidden) # [B, Li]

            logits[:, :, step] = logits_att

            # Get maximum probabilities and indices
            indices = torch.argmax(logits_att, dim=1, keepdim=True) # [B, 1]
            decoder_input = torch.take_along_dim(embedded_inputs, indices.unsqueeze(-1), dim=1)
            pointers[:, step] = indices[:, 0]

        return (logits, pointers), hidden

class PointerNet(nn.Module):
    """
    PointerNet taking as input a tensor of shape [B, Li, C]
    and that outputs a sequence of :
        - Log probabilities [B, Lo, Li] where each [b, l, :] is a distribution over the input sequence
        - Pointers [B, Lo]
    """
    def __init__(self,
                 input_dim:int,
                 output_length:int,
                 embedding_dim:int,
                 hidden_dim:int,
                 lstm_layers:int,
                 dropout:float=0.,
                 bidir:bool=False):
        """
        input_dim (C): Data input dimension
        output_length (Lo): Lenght of output sequence
        embedding_dim (E): Number of embbeding channels
        hidden_dim (H): Encoders hidden units
        lstm_layers (N): Number of layers for LSTMs
        dropout: Float between 0-1
        bidir: Bidirectional
        """
        super(PointerNet, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.lstm_layers = lstm_layers
        self.name = "PointerNet"

        self.embedding = nn.Linear(self.input_dim, embedding_dim)

        self.encoder = Encoder(embedding_dim,
                               hidden_dim,
                               lstm_layers,
                               dropout,
                               bidir)
    
        self.decoder = Decoder(embedding_dim,
                               hidden_dim,
                               lstm_layers,
                               output_length)
        
        self.decoder_input0 = Parameter(torch.Tensor(embedding_dim), requires_grad=False)
        torch.nn.init.uniform_(self.decoder_input0)

        self.pointers = None

    def forward(self, inputs):
        """
        Inputs:
            - inputs: Input sequence [B, Li, C]
        Returns:
            - logits probabilities [B, Li, Lo]
        """
        B = inputs.shape[0]

        embedded_inputs = self.embedding(inputs)

        encoder_hiddens, hidden  = self.encoder(embedded_inputs)

        if self.bidir:
            hidden = (hidden[0].reshape(self.lstm_layers, B, -1), hidden[1].reshape(self.lstm_layers, B, -1))

        decoder_input0 = self.decoder_input0.expand(B, -1).unsqueeze(1)
        (logits, self.pointers), _ = self.decoder(embedded_inputs, decoder_input0, hidden, encoder_hiddens)

        return logits
    
    def select(self, inputs):
        """
        Inputs:
            - inputs: Input sequence [B, Li, C]
        Returns:
            - outputs: elements from the input sequence [B, Lo, C]
        """
        logits = self.forward(inputs)
        max_probs = torch.max(torch.softmax(logits, dim=1), dim=1)[0]
        outputs = torch.take_along_dim(inputs, self.pointers.unsqueeze(-1), dim=1)
        return outputs, max_probs

class PointerNetHidden(nn.Module):
    """
    PointerNet taking as input a tensor of shape [B, Li, C]
    and that outputs a sequence of :
        - Log probabilities [B, Lo, Li] where each [b, l, :] is a distribution over the input sequence
        - Pointers [B, Lo]
    """
    def __init__(self,
                 input_dim:int,
                 output_length:int,
                 exclude_first:int,
                 hidden_dim:int,
                 lstm_layers:int,
                 dropout:float=0.,
                 bidir:bool=False):
        """
        input_dim (C): Data input dimension
        output_length (Lo): Lenght of output sequence
        embedding_dim (E): Number of embbeding channels
        hidden_dim (H): Encoders hidden units
        lstm_layers (N): Number of layers for LSTMs
        dropout: Float between 0-1
        bidir: Bidirectional
        """
        super(PointerNetHidden, self).__init__()
        self.input_dim = input_dim
        self.bidir = bidir
        self.exclude_first = exclude_first
        self.lstm_layers = lstm_layers
        self.name = "PointerNetHidden"

        self.embedding_hidden = nn.Sequential(*[
            nn.Linear(self.input_dim * self.exclude_first, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2 if self.bidir else hidden_dim)
            ])

        self.encoder = Encoder(self.input_dim,
                               hidden_dim,
                               lstm_layers,
                               dropout,
                               bidir)
    
        self.decoder = Decoder(self.input_dim,
                               hidden_dim,
                               lstm_layers,
                               output_length)
        
        self.decoder_input0 = Parameter(torch.Tensor(self.input_dim), requires_grad=False)
        torch.nn.init.uniform_(self.decoder_input0)

        self.pointers = None

    def forward(self, inputs):
        """
        Inputs:
            - inputs: Input sequence [B, Li, C]
        Returns:
            - logits probabilities [B, Li, Lo]
        """
        B = inputs.shape[0]

        h0 = self.embedding_hidden(inputs[:, :self.exclude_first, :].reshape((1, B, -1)))
        h0 = torch.tile(h0, (self.lstm_layers if not(self.bidir) else self.lstm_layers * 2, 1, 1))
        c0 = torch.zeros_like(h0)

        encoder_hiddens, hidden  = self.encoder(inputs[:, self.exclude_first:, :], (h0, c0))

        if self.bidir:
            hidden = (hidden[0].reshape(self.lstm_layers, B, -1), hidden[1].reshape(self.lstm_layers, B, -1))

        decoder_input0 = self.decoder_input0.expand(B, -1).unsqueeze(1)
        (logits, self.pointers), _ = self.decoder(inputs[:, self.exclude_first:, :], decoder_input0, hidden, encoder_hiddens)
        logits = torch.cat((torch.zeros((B, self.exclude_first, logits.shape[-1])),
                              logits), dim=1)
        self.pointers += self.exclude_first
        return logits
    
    def select(self, inputs):
        """
        Inputs:
            - inputs: Input sequence [B, Li, C]
        Returns:
            - outputs: elements from the input sequence [B, Lo, C]
        """
        logits = self.forward(inputs)
        max_probs = torch.max(torch.softmax(logits, dim=1), dim=1)[0]
        outputs = torch.take_along_dim(inputs, self.pointers.unsqueeze(-1), dim=1)
        return outputs, max_probs