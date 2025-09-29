import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.q_linear = nn.Linear(input_dim, input_dim)
        self.k_linear = nn.Linear(input_dim, input_dim)
        self.v_linear = nn.Linear(input_dim, input_dim)
        self.output_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x shape: batch_size x seq_len x input_dim

        batch_size, seq_len, input_dim = x.size()

        # Project the input vectors to queries, keys, and values
        queries = self.q_linear(x).view(batch_size, seq_len, self.num_heads, input_dim // self.num_heads).transpose(1, 2)
        keys = self.k_linear(x).view(batch_size, seq_len, self.num_heads, input_dim // self.num_heads).transpose(1, 2)
        values = self.v_linear(x).view(batch_size, seq_len, self.num_heads, input_dim // self.num_heads).transpose(1, 2)

        # Compute the dot product of queries and keys
        dot_product = torch.matmul(queries, keys.transpose(-2, -1)) / (input_dim // self.num_heads) ** 0.5

        # Apply the softmax function to obtain attention weights
        attention_weights = torch.softmax(dot_product, dim=-1)

        # Compute the weighted sum of values
        weighted_sum = torch.matmul(attention_weights, values)

        # Reshape the output and apply a linear transformation
        weighted_sum = weighted_sum.transpose(1, 2).contiguous().view(batch_size, seq_len, input_dim)
        output = self.output_linear(weighted_sum)

        return output

