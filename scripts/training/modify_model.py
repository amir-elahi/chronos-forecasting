from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5LayerSelfAttention
import os

# TODO: Check if the weights are copied correctly: Done but double check
# TODO: Make it work for all the amazon model sizes: Done but double check
# TODO: Make it work for the T5 models
# TODO: Make it work for random initialization of weights.
# !Make sure that you change the T5ForConditionalGeneration code before starting the fine-tuning process.


# Define a new layer type for Channel Attention
class ChannelAttentionLayer(T5LayerSelfAttention):
    def __init__(self, config, batch_size, num_channels, embedding_dim=512):
        super().__init__(config)

        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.batch_size = batch_size

    def forward(self, hidden_states):
        # input shape [batch_size*num_channels, context_length, dim]
        # print('Hidden state shape input',hidden_states.shape)
        hidden_states = hidden_states.reshape(self.batch_size, self.num_channels, -1, self.embedding_dim)  # [batch_size, num_channels , context_length, dim]
        hidden_states = hidden_states.permute(0, 2, 1, 3)  # [batch_size, context_length, num_channels, dim]
        hidden_states = hidden_states.reshape(-1, self.num_channels, self.embedding_dim)  # [batch_size*context_length, num_channels, dim]
        # print('Hidden state shape after first view',hidden_states.shape)
        output = super().forward(hidden_states)
        hidden_states = output[0]  # [batch_size*context_length, num_channels, dim]
        # print('Hidden state shape after super forward',hidden_states.shape)
        hidden_states = hidden_states.reshape(self.batch_size, self.num_channels, -1, self.embedding_dim)  # [batch_size, num_channels, context_length, dim]
        hidden_states = hidden_states.permute(0, 2, 1, 3)  # [batch_size, context_length, num_channels, dim]
        hidden_states = hidden_states.reshape(self.batch_size * self.num_channels, -1, self.embedding_dim)  # [batch_size*num_channels, context_length, dim]
        # print('Hidden state shape after second view',hidden_states.shape)
        output = (hidden_states,) + output[1:]  # Add the rest of the outputs
        return output


class CustomT5(T5ForConditionalGeneration):
    def __init__(self, config, batch_size, num_channels, embedding_dim=512):
        super().__init__(config)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim

        for i in range(len(self.encoder.block)):
            encoder_block = self.encoder.block[i]

            # Create a channel attention layer with a distinct class
            channel_attention = ChannelAttentionLayer(config, self.batch_size, self.num_channels, self.embedding_dim)

            # Insert the channel attention layer after the self_attention layer in the list
            encoder_block.layer.insert(1, channel_attention)

    @classmethod
    def from_pretrained(cls, model_id: str = "amazon/chronos-t5-small", **kwargs):

        num_channels = kwargs.pop("num_channels", 3)
        batch_size = kwargs.pop("batch_size", 1)
        Multivariate = kwargs.pop("Multivariate", True)

        # Load the model and its configuration
        original_model = T5ForConditionalGeneration.from_pretrained(model_id)
        embedding_dim = original_model.shared.embedding_dim
        config = original_model.config

        # Instantiate the model with default parameters (for now)
        model = cls(config, batch_size=batch_size, num_channels=num_channels, embedding_dim=embedding_dim)  # Default values, to be overridden later
        model.config.MultivariateConfig = {'batch_size': batch_size, 'num_channels': num_channels, 'Mutivariate': Multivariate}

        # Create a new state dictionary for the custom model
        custom_state_dict = {}

        # Modify the state dictionary from the copied model since the order of layers has changed
        for key, value in original_model.state_dict().items():
            # Rename keys as necessary
            if "encoder.block." in key:
                # Check if the layer index needs adjustment
                if "layer.1.DenseReluDense" in key:  # If it is the layer we added
                    new_key = key.replace("layer.1.DenseReluDense", "layer.2.DenseReluDense")  # Shift layer 1 to layer 2
                elif "layer.1.layer_norm" in key:
                    new_key = key.replace("layer.1.layer_norm", "layer.2.layer_norm")
                elif "layer.0.SelfAttention" in key:
                    added_key = key.replace("layer.0.SelfAttention", "layer.1.SelfAttention")
                    custom_state_dict[added_key] = value  # Add the new key
                    new_key = key
                elif "layer.0.layer_norm" in key:
                    added_key = key.replace("layer.0.layer_norm", "layer.1.layer_norm")
                    custom_state_dict[added_key] = value  # Add the new key
                    new_key = key
                else:
                    new_key = key  # No change needed for other layers
            else:
                new_key = key  # No change needed

            custom_state_dict[new_key] = value  # Add the new key

            model.load_state_dict(custom_state_dict, strict=False)

        return model

    def require_grad(self, channel_attention=True, Rest=False):

        for param in self.parameters():
            param.requires_grad = False

        if channel_attention and Rest:
            for param in self.parameters():
                param.requires_grad = True
        elif channel_attention and not Rest:
            for block in self.encoder.block:
                for param in block.layer[1].parameters():
                    param.requires_grad = True
        elif not channel_attention and Rest:
            for param in self.parameters():
                param.requires_grad = True
            for block in self.encoder.block:
                for param in block.layer[1].parameters():
                    param.requires_grad = False
        else:
            Warning("No parameters are set to require grad")


if __name__ == "__main__":
    # Load the model
    model = CustomT5.from_pretrained(model_id="amazon/chronos-t5-small", num_channels=3, batch_size=1)
    model.require_grad(channel_attention=True, Rest=False)
    original_model = T5ForConditionalGeneration.from_pretrained("amazon/chronos-t5-small")
