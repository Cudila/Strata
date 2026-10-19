# Nemotron 3.5 Super VL: карта тензоров языковой модели

Сгенерировано `tools/nemotron/tensor_map.py` из заголовков safetensors: NVFP4.
Слои `backbone` нумеруются 0..87, тип слоя из `layers_block_type`; `mtp.layers.0` — attention-подслой MTP, `mtp.layers.1` — его MoE-подслой. `E` — номер эксперта 0..511. Объём — по всем вхождениям шаблона.

Порядок блоков: `MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME`

| Шаблон | Тип слоя | Слои | Кол-во | NVFP4: dtype, форма, объём |
|---|---|---|---|---|
| `backbone.layers.L.mixer.A_log` | mamba | 40 layers (0..86) | 40 | BF16 [128] 0.0 MiB |
| `backbone.layers.L.mixer.D` | mamba | 40 layers (0..86) | 40 | BF16 [128] 0.0 MiB |
| `backbone.layers.L.mixer.conv1d.bias` | mamba | 40 layers (0..86) | 40 | BF16 [10240] 0.8 MiB |
| `backbone.layers.L.mixer.conv1d.weight` | mamba | 40 layers (0..86) | 40 | BF16 [10240, 1, 4] 3.1 MiB |
| `backbone.layers.L.mixer.dt_bias` | mamba | 40 layers (0..86) | 40 | BF16 [128] 0.0 MiB |
| `backbone.layers.L.mixer.experts.E.down_proj.input_scale` | moe | 40 layers (1..87) | 20480 | F32 [] 0.1 MiB |
| `backbone.layers.L.mixer.experts.E.down_proj.weight` | moe | 40 layers (1..87) | 20480 | U8 [1024, 1344] 26880.0 MiB |
| `backbone.layers.L.mixer.experts.E.down_proj.weight_scale` | moe | 40 layers (1..87) | 20480 | F8_E4M3 [1024, 168] 3360.0 MiB |
| `backbone.layers.L.mixer.experts.E.down_proj.weight_scale_2` | moe | 40 layers (1..87) | 20480 | F32 [] 0.1 MiB |
| `backbone.layers.L.mixer.experts.E.up_proj.input_scale` | moe | 40 layers (1..87) | 20480 | F32 [] 0.1 MiB |
| `backbone.layers.L.mixer.experts.E.up_proj.weight` | moe | 40 layers (1..87) | 20480 | U8 [2688, 512] 26880.0 MiB |
| `backbone.layers.L.mixer.experts.E.up_proj.weight_scale` | moe | 40 layers (1..87) | 20480 | F8_E4M3 [2688, 64] 3360.0 MiB |
| `backbone.layers.L.mixer.experts.E.up_proj.weight_scale_2` | moe | 40 layers (1..87) | 20480 | F32 [] 0.1 MiB |
| `backbone.layers.L.mixer.fc1_latent_proj.weight` | moe | 40 layers (1..87) | 40 | BF16 [1024, 4096] 320.0 MiB |
| `backbone.layers.L.mixer.fc2_latent_proj.weight` | moe | 40 layers (1..87) | 40 | BF16 [4096, 1024] 320.0 MiB |
| `backbone.layers.L.mixer.gate.e_score_correction_bias` | moe | 40 layers (1..87) | 40 | F32 [512] 0.1 MiB |
| `backbone.layers.L.mixer.gate.weight` | moe | 40 layers (1..87) | 40 | BF16 [512, 4096] 160.0 MiB |
| `backbone.layers.L.mixer.in_proj.input_scale` | mamba | 40 layers (0..86) | 40 | F32 [1] 0.0 MiB |
| `backbone.layers.L.mixer.in_proj.weight` | mamba | 40 layers (0..86) | 40 | F8_E4M3 [18560, 4096] 2900.0 MiB |
| `backbone.layers.L.mixer.in_proj.weight_scale` | mamba | 40 layers (0..86) | 40 | F32 [] 0.0 MiB |
| `backbone.layers.L.mixer.k_proj.input_scale` | attention | 7,16,25,36,47,58,69,78 | 8 | F32 [1] 0.0 MiB |
| `backbone.layers.L.mixer.k_proj.weight` | attention | 7,16,25,36,47,58,69,78 | 8 | F8_E4M3 [256, 4096] 8.0 MiB |
| `backbone.layers.L.mixer.k_proj.weight_scale` | attention | 7,16,25,36,47,58,69,78 | 8 | F32 [] 0.0 MiB |
| `backbone.layers.L.mixer.norm.weight` | mamba | 40 layers (0..86) | 40 | BF16 [8192] 0.6 MiB |
| `backbone.layers.L.mixer.o_proj.input_scale` | attention | 7,16,25,36,47,58,69,78 | 8 | F32 [1] 0.0 MiB |
| `backbone.layers.L.mixer.o_proj.weight` | attention | 7,16,25,36,47,58,69,78 | 8 | F8_E4M3 [4096, 4096] 128.0 MiB |
| `backbone.layers.L.mixer.o_proj.weight_scale` | attention | 7,16,25,36,47,58,69,78 | 8 | F32 [] 0.0 MiB |
| `backbone.layers.L.mixer.out_proj.input_scale` | mamba | 40 layers (0..86) | 40 | F32 [1] 0.0 MiB |
| `backbone.layers.L.mixer.out_proj.weight` | mamba | 40 layers (0..86) | 40 | F8_E4M3 [4096, 8192] 1280.0 MiB |
| `backbone.layers.L.mixer.out_proj.weight_scale` | mamba | 40 layers (0..86) | 40 | F32 [] 0.0 MiB |
| `backbone.layers.L.mixer.q_proj.input_scale` | attention | 7,16,25,36,47,58,69,78 | 8 | F32 [1] 0.0 MiB |
| `backbone.layers.L.mixer.q_proj.weight` | attention | 7,16,25,36,47,58,69,78 | 8 | F8_E4M3 [4096, 4096] 128.0 MiB |
| `backbone.layers.L.mixer.q_proj.weight_scale` | attention | 7,16,25,36,47,58,69,78 | 8 | F32 [] 0.0 MiB |
| `backbone.layers.L.mixer.shared_experts.down_proj.input_scale` | moe | 40 layers (1..87) | 40 | F32 [] 0.0 MiB |
| `backbone.layers.L.mixer.shared_experts.down_proj.weight` | moe | 40 layers (1..87) | 40 | U8 [4096, 2688] 420.0 MiB |
| `backbone.layers.L.mixer.shared_experts.down_proj.weight_scale` | moe | 40 layers (1..87) | 40 | F8_E4M3 [4096, 336] 52.5 MiB |
| `backbone.layers.L.mixer.shared_experts.down_proj.weight_scale_2` | moe | 40 layers (1..87) | 40 | F32 [] 0.0 MiB |
| `backbone.layers.L.mixer.shared_experts.up_proj.input_scale` | moe | 40 layers (1..87) | 40 | F32 [] 0.0 MiB |
| `backbone.layers.L.mixer.shared_experts.up_proj.weight` | moe | 40 layers (1..87) | 40 | U8 [5376, 2048] 420.0 MiB |
| `backbone.layers.L.mixer.shared_experts.up_proj.weight_scale` | moe | 40 layers (1..87) | 40 | F8_E4M3 [5376, 256] 52.5 MiB |
| `backbone.layers.L.mixer.shared_experts.up_proj.weight_scale_2` | moe | 40 layers (1..87) | 40 | F32 [] 0.0 MiB |
| `backbone.layers.L.mixer.v_proj.input_scale` | attention | 7,16,25,36,47,58,69,78 | 8 | F32 [1] 0.0 MiB |
| `backbone.layers.L.mixer.v_proj.weight` | attention | 7,16,25,36,47,58,69,78 | 8 | F8_E4M3 [256, 4096] 8.0 MiB |
| `backbone.layers.L.mixer.v_proj.weight_scale` | attention | 7,16,25,36,47,58,69,78 | 8 | F32 [] 0.0 MiB |
| `backbone.layers.L.norm.weight` | attention/mamba/moe | 0-87 | 88 | BF16 [4096] 0.7 MiB |
| `mtp.layers.L.eh_proj.weight` | mtp-attention | 0 | 1 | BF16 [4096, 8192] 64.0 MiB |
| `mtp.layers.L.enorm.weight` | mtp-attention | 0 | 1 | BF16 [4096] 0.0 MiB |
| `mtp.layers.L.final_layernorm.weight` | mtp-moe | 1 | 1 | BF16 [4096] 0.0 MiB |
| `mtp.layers.L.hnorm.weight` | mtp-attention | 0 | 1 | BF16 [4096] 0.0 MiB |
| `mtp.layers.L.mixer.experts.E.down_proj.input_scale` | mtp-moe | 1 | 512 | F32 [1] 0.0 MiB |
| `mtp.layers.L.mixer.experts.E.down_proj.weight` | mtp-moe | 1 | 512 | U8 [1024, 1344] 672.0 MiB |
| `mtp.layers.L.mixer.experts.E.down_proj.weight_scale` | mtp-moe | 1 | 512 | F8_E4M3 [1024, 168] 84.0 MiB |
| `mtp.layers.L.mixer.experts.E.down_proj.weight_scale_2` | mtp-moe | 1 | 512 | F32 [1] 0.0 MiB |
| `mtp.layers.L.mixer.experts.E.up_proj.input_scale` | mtp-moe | 1 | 512 | F32 [1] 0.0 MiB |
| `mtp.layers.L.mixer.experts.E.up_proj.weight` | mtp-moe | 1 | 512 | U8 [2688, 512] 672.0 MiB |
| `mtp.layers.L.mixer.experts.E.up_proj.weight_scale` | mtp-moe | 1 | 512 | F8_E4M3 [2688, 64] 84.0 MiB |
| `mtp.layers.L.mixer.experts.E.up_proj.weight_scale_2` | mtp-moe | 1 | 512 | F32 [1] 0.0 MiB |
| `mtp.layers.L.mixer.fc1_latent_proj.weight` | mtp-moe | 1 | 1 | BF16 [1024, 4096] 8.0 MiB |
| `mtp.layers.L.mixer.fc2_latent_proj.weight` | mtp-moe | 1 | 1 | BF16 [4096, 1024] 8.0 MiB |
| `mtp.layers.L.mixer.gate.e_score_correction_bias` | mtp-moe | 1 | 1 | F32 [512] 0.0 MiB |
| `mtp.layers.L.mixer.gate.weight` | mtp-moe | 1 | 1 | BF16 [512, 4096] 4.0 MiB |
| `mtp.layers.L.mixer.k_proj.k_scale` | mtp-attention | 0 | 1 | F32 [1] 0.0 MiB |
| `mtp.layers.L.mixer.k_proj.weight` | mtp-attention | 0 | 1 | BF16 [256, 4096] 2.0 MiB |
| `mtp.layers.L.mixer.o_proj.weight` | mtp-attention | 0 | 1 | BF16 [4096, 4096] 32.0 MiB |
| `mtp.layers.L.mixer.q_proj.weight` | mtp-attention | 0 | 1 | BF16 [4096, 4096] 32.0 MiB |
| `mtp.layers.L.mixer.shared_experts.down_proj.input_scale` | mtp-moe | 1 | 1 | F32 [1] 0.0 MiB |
| `mtp.layers.L.mixer.shared_experts.down_proj.weight` | mtp-moe | 1 | 1 | U8 [4096, 2688] 10.5 MiB |
| `mtp.layers.L.mixer.shared_experts.down_proj.weight_scale` | mtp-moe | 1 | 1 | F8_E4M3 [4096, 336] 1.3 MiB |
| `mtp.layers.L.mixer.shared_experts.down_proj.weight_scale_2` | mtp-moe | 1 | 1 | F32 [1] 0.0 MiB |
| `mtp.layers.L.mixer.shared_experts.up_proj.input_scale` | mtp-moe | 1 | 1 | F32 [1] 0.0 MiB |
| `mtp.layers.L.mixer.shared_experts.up_proj.weight` | mtp-moe | 1 | 1 | U8 [5376, 2048] 10.5 MiB |
| `mtp.layers.L.mixer.shared_experts.up_proj.weight_scale` | mtp-moe | 1 | 1 | F8_E4M3 [5376, 256] 1.3 MiB |
| `mtp.layers.L.mixer.shared_experts.up_proj.weight_scale_2` | mtp-moe | 1 | 1 | F32 [1] 0.0 MiB |
| `mtp.layers.L.mixer.v_proj.v_scale` | mtp-attention | 0 | 1 | F32 [1] 0.0 MiB |
| `mtp.layers.L.mixer.v_proj.weight` | mtp-attention | 0 | 1 | BF16 [256, 4096] 2.0 MiB |
| `mtp.layers.L.norm.weight` | mtp-both | 0-1 | 2 | BF16 [4096] 0.0 MiB |
| `backbone.embeddings.weight` | global | - | 1 | BF16 [131072, 4096] 1024.0 MiB |
| `backbone.norm_f.weight` | global | - | 1 | BF16 [4096] 0.0 MiB |
| `lm_head.input_scale` | global | - | 1 | F32 [1] 0.0 MiB |
| `lm_head.weight` | global | - | 1 | F8_E4M3 [131072, 4096] 512.0 MiB |
| `lm_head.weight_scale` | global | - | 1 | F32 [] 0.0 MiB |
