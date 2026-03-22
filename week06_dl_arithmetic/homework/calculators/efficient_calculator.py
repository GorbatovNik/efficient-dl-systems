from calculators.base import BaseCalculator


class EfficientCalculator(BaseCalculator):

    def calculate_total_params(self) -> int:
        V = self.model.vocab_size
        H = self.model.hidden_dim
        I = self.model.intermediate_dim
        L = self.model.num_layers

        embedding = V * H
        per_layer = (
            3 * H * H          # fused qkv_proj
            + H * H            # out_proj
            + 2 * H * I        # gate_proj, up_proj
            + I * H            # down_proj
            + 2 * H            # 2x RMSNorm weight
        )
        final_norm = H
        lm_head = V * H

        return embedding + L * per_layer + final_norm + lm_head

    def calculate_param_memory(self) -> int:
        total = self.calculate_total_params()
        return total * 4 // self.training.num_gpus

    def calculate_gradient_memory(self) -> int:
        total = self.calculate_total_params()
        return total * 4 // self.training.num_gpus

    def calculate_optimizer_memory(self) -> int:
        total = self.calculate_total_params()
        return 3 * total * 4 // self.training.num_gpus

    def calculate_fsdp_buffer_memory(self) -> int:
        H = self.model.hidden_dim
        I = self.model.intermediate_dim
        d = self.training.dtype_bytes

        largest_layer_params = (
            3 * H * H
            + H * H
            + 2 * H * I
            + I * H
            + 2 * H
        )
        allgather_buffers = 2 * largest_layer_params * d
        reducescatter_buffers = 2 * largest_layer_params * d
        return allgather_buffers + reducescatter_buffers

    def calculate_activation_memory(self) -> int:
        B = self.training.batch_size
        S = self.training.seq_len
        H = self.model.hidden_dim
        I = self.model.intermediate_dim
        L = self.model.num_layers
        d = self.training.dtype_bytes

        embedding_act = B * S * H * d

        per_layer_rmsnorm = 2 * (
            B * S * H * d
            + B * S * 1 * 4
        )

        per_layer_attn = (
            B * S * H * d
            + B * S * H * d
            + B * S * H * d
            + B * S * H * d
            + B * S * H * d
            + B * S * H * d
        )

        per_layer_mlp = B * S * H * d

        per_layer_residual = 2 * B * S * H * d

        per_layer = per_layer_rmsnorm + per_layer_attn + per_layer_mlp + per_layer_residual

        final_norm = B * S * H * d + B * S * 1 * 4

        return embedding_act + L * per_layer + final_norm

    def calculate_peak_memory(self) -> int:
        return (
            self.calculate_param_memory()
            + self.calculate_gradient_memory()
            + self.calculate_optimizer_memory()
            + self.calculate_fsdp_buffer_memory()
            + self.calculate_activation_memory()
        )

    def time_embedding_ms(self) -> float:
        B = self.training.batch_size
        S = self.training.seq_len
        H = self.model.hidden_dim
        V = self.model.vocab_size
        d = self.training.dtype_bytes
        flops = 0
        memory_bytes = B * S * 4 + B * S * H * d + V * H * d
        return self.roofline_time_ms(flops, memory_bytes)

    def time_rms_norm_ms(self) -> float:
        B = self.training.batch_size
        S = self.training.seq_len
        H = self.model.hidden_dim
        d = self.training.dtype_bytes
        flops = B * S * H * 5
        memory_bytes = 2 * B * S * H * d + H * d
        return self.roofline_time_ms(flops, memory_bytes)

    def time_attention_ms(self) -> float:
        B = self.training.batch_size
        S = self.training.seq_len
        H = self.model.hidden_dim
        n_heads = self.model.num_heads
        head_dim = H // n_heads
        d = self.training.dtype_bytes

        qkv_flops = 2 * B * S * H * 3 * H
        qkv_mem = B * S * H * d + B * S * 3 * H * d + H * 3 * H * d

        fa_flops = 2 * 2 * B * n_heads * S * S * head_dim
        fa_mem = 3 * B * S * H * d + B * S * H * d

        out_flops = 2 * B * S * H * H
        out_mem = 2 * B * S * H * d + H * H * d

        total_flops = qkv_flops + fa_flops + out_flops
        total_mem = qkv_mem + fa_mem + out_mem
        return self.roofline_time_ms(total_flops, total_mem)

    def time_mlp_ms(self) -> float:
        B = self.training.batch_size
        S = self.training.seq_len
        H = self.model.hidden_dim
        I = self.model.intermediate_dim
        d = self.training.dtype_bytes

        gate_up_flops = 2 * 2 * B * S * H * I
        gate_up_mem = 2 * (B * S * H * d + B * S * I * d + H * I * d)

        activation_flops = B * S * I * 10
        activation_mem = 2 * B * S * I * d

        down_flops = 2 * B * S * I * H
        down_mem = B * S * I * d + B * S * H * d + I * H * d

        total_flops = gate_up_flops + activation_flops + down_flops
        total_mem = gate_up_mem + activation_mem + down_mem
        return self.roofline_time_ms(total_flops, total_mem)

    def time_lm_head_ms(self) -> float:
        return 0.0

    def time_loss_ms(self) -> float:
        B = self.training.batch_size
        S = self.training.seq_len
        H = self.model.hidden_dim
        V = self.model.vocab_size
        d = self.training.dtype_bytes

        flops = 2 * B * S * H * V + B * S * V * 5
        memory_bytes = B * S * H * d + H * V * d + B * S * 4
        return self.roofline_time_ms(flops, memory_bytes)

    def calculate_allgather_volume(self) -> int:
        total = self.calculate_total_params()
        return total * 2

    def calculate_reducescatter_volume(self) -> int:
        total = self.calculate_total_params()
        return total * 2

    def calculate_communication_volume(self) -> int:
        return 2 * self.calculate_allgather_volume() + self.calculate_reducescatter_volume()

    def time_communication_ms(self) -> float:
        volume = self.calculate_communication_volume()
        return volume / (self.gpu.interconnect_bandwidth_gbps * 1e9) * 1000

    def overlap_efficiency(self) -> float:
        return 0.9

    def time_total_step_ms(self) -> float:
        compute = self.time_forward_backward_ms()
        comm = self.time_communication_ms()
        overlap = self.overlap_efficiency()
        return compute + (1.0 - overlap) * comm
