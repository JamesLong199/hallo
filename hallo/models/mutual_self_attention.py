# pylint: disable=E1120
"""
This module contains the implementation of mutual self-attention, 
which is a type of attention mechanism used in deep learning models. 
The module includes several classes and functions related to attention mechanisms, 
such as BasicTransformerBlock and TemporalBasicTransformerBlock. 
The main purpose of this module is to provide a comprehensive attention mechanism for various tasks in deep learning, 
such as image and video processing, natural language processing, and so on.
"""

from typing import Any, Dict, Optional

import torch
from einops import rearrange

from .attention import BasicTransformerBlock, TemporalBasicTransformerBlock
import time
from torch.profiler import profile, record_function, ProfilerActivity

import math

def torch_dfs(model: torch.nn.Module):
    """
    Perform a depth-first search (DFS) traversal on a PyTorch model's neural network architecture.

    This function recursively traverses all the children modules of a given PyTorch model and returns a list
    containing all the modules in the model's architecture. The DFS approach starts with the input model and
    explores its children modules depth-wise before backtracking and exploring other branches.

    Args:
        model (torch.nn.Module): The root module of the neural network to traverse.

    Returns:
        list: A list of all the modules in the model's architecture.
    """
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class ReferenceAttentionControl:
    """
    This class is used to control the reference attention mechanism in a neural network model.
    It is responsible for managing the guidance and fusion blocks, and modifying the self-attention
    and group normalization mechanisms. The class also provides methods for registering reference hooks
    and updating/clearing the internal state of the attention control object.

    Attributes:
        unet: The UNet model associated with this attention control object.
        mode: The operating mode of the attention control object, either 'write' or 'read'.
        do_classifier_free_guidance: Whether to use classifier-free guidance in the attention mechanism.
        attention_auto_machine_weight: The weight assigned to the attention auto-machine.
        gn_auto_machine_weight: The weight assigned to the group normalization auto-machine.
        style_fidelity: The style fidelity parameter for the attention mechanism.
        reference_attn: Whether to use reference attention in the model.
        reference_adain: Whether to use reference AdaIN in the model.
        fusion_blocks: The type of fusion blocks to use in the model ('midup', 'late', or 'nofusion').
        batch_size: The batch size used for processing video frames.

    Methods:
        register_reference_hooks: Registers the reference hooks for the attention control object.
        hacked_basic_transformer_inner_forward: The modified inner forward method for the basic transformer block.
        update: Updates the internal state of the attention control object using the provided writer and dtype.
        clear: Clears the internal state of the attention control object.
    """
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        attention_auto_machine_weight=float("inf"),
        gn_auto_machine_weight=1.0,
        style_fidelity=1.0,
        reference_attn=True,
        reference_adain=False,
        fusion_blocks="midup",
        batch_size=1,
        # face_masks=None,
    ) -> None:
        """
       Initializes the ReferenceAttentionControl class.

       Args:
           unet (torch.nn.Module): The UNet model.
           mode (str, optional): The mode of operation. Defaults to "write".
           do_classifier_free_guidance (bool, optional): Whether to do classifier-free guidance. Defaults to False.
           attention_auto_machine_weight (float, optional): The weight for attention auto-machine. Defaults to infinity.
           gn_auto_machine_weight (float, optional): The weight for group-norm auto-machine. Defaults to 1.0.
           style_fidelity (float, optional): The style fidelity. Defaults to 1.0.
           reference_attn (bool, optional): Whether to use reference attention. Defaults to True.
           reference_adain (bool, optional): Whether to use reference AdaIN. Defaults to False.
           fusion_blocks (str, optional): The fusion blocks to use. Defaults to "midup".
           batch_size (int, optional): The batch size. Defaults to 1.

       Raises:
           ValueError: If the mode is not recognized.
           ValueError: If the fusion blocks are not recognized.
       """
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            fusion_blocks,
            batch_size=batch_size,
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        _attention_auto_machine_weight,
        _gn_auto_machine_weight,
        _style_fidelity,
        _reference_attn,
        _reference_adain,
        _dtype=torch.float16,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cpu"),
        _fusion_blocks="midup",
    ):
        """
        Registers reference hooks for the model.

        This function is responsible for registering reference hooks in the model, 
        which are used to modify the attention mechanism and group normalization layers.
        It takes various parameters as input, such as mode, 
        do_classifier_free_guidance, _attention_auto_machine_weight, _gn_auto_machine_weight, _style_fidelity,
        _reference_attn, _reference_adain, _dtype, batch_size, num_images_per_prompt, device, and _fusion_blocks.

        Args:
            self: Reference to the instance of the class.
            mode: The mode of operation for the reference hooks.
            do_classifier_free_guidance: A boolean flag indicating whether to use classifier-free guidance.
            _attention_auto_machine_weight: The weight for the attention auto-machine.
            _gn_auto_machine_weight: The weight for the group normalization auto-machine.
            _style_fidelity: The style fidelity for the reference hooks.
            _reference_attn: A boolean flag indicating whether to use reference attention.
            _reference_adain: A boolean flag indicating whether to use reference AdaIN.
            _dtype: The data type for the reference hooks.
            batch_size: The batch size for the reference hooks.
            num_images_per_prompt: The number of images per prompt for the reference hooks.
            device: The device for the reference hooks.
            _fusion_blocks: The fusion blocks for the reference hooks.

        Returns:
            None
        """
        MODE = mode
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length=None,
            block_name='unknown_block',
            mask_fg=None,
        ):
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            gate_msa = None
            shift_mlp = None
            scale_mlp = None
            gate_mlp = None
            start_event.record()
            if self.use_ada_layer_norm:  # False
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                (
                    norm_hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            end_event.record()
            end_event.synchronize()
            time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
            print(f"mutual_self_attention norm1 took {time_elapsed} seconds", flush=True)
            # 1. Self-Attention
            # self.only_cross_attention = False
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=(
                        encoder_hidden_states if self.only_cross_attention else None
                    ),
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    self.bank.append(norm_hidden_states.clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=(
                            encoder_hidden_states if self.only_cross_attention else None
                        ),
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if MODE == "read":
                    start_event.record()
                    print(f"mutual_self_attention 245 block_name: {block_name}", flush=True)
                    bank_fea = [
                        rearrange(
                            rearrange(
                                d,
                                "(b s) l c -> b s l c",
                                b=norm_hidden_states.shape[0] // video_length,
                            )[:, 0, :, :]
                            # .unsqueeze(1)
                            .repeat(1, video_length, 1, 1),
                            "b t l c -> (b t) l c",
                        )
                        for d in self.bank
                    ]
                    motion_frames_fea = [rearrange(
                        d,
                        "(b s) l c -> b s l c",
                        b=norm_hidden_states.shape[0] // video_length,
                    )[:, 1:, :, :] for d in self.bank]
                    
                    # # if 'upblock_2' in block_name or 'upblock_3' in block_name or 'downblock_0_layer_1' in block_name:
                    # if 'upblock_2' in block_name:
                    # # if 'upblock_2' in block_name or 'upblock_3' in block_name:
                    # # if 'downblock_0_layer_1' in block_name:
                    #     modify_norm_hidden_states = norm_hidden_states
                    # else:
                    #     modify_norm_hidden_states = torch.cat(
                    #         [norm_hidden_states] + bank_fea, dim=1
                    #     )
                    modify_norm_hidden_states = torch.cat(
                            [norm_hidden_states] + bank_fea, dim=1
                        )
                    end_event.record()
                    end_event.synchronize()
                    time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
                    print(f"mutual_self_attention before attn1 took {time_elapsed} seconds", flush=True)
                    print(f"mutual_self_attention attn1 modify_norm_hidden_states shape: {modify_norm_hidden_states.shape}", flush=True)
                    start_event.record()
                    # print('mutual_self_attention attn1 residual_connection:', self.attn1.residual_connection, flush=True)
                    
                    # if mask_fg is not None:
                    #     print(f"mutual_self_attention attn1 mask_fg is not None:", flush=True)
                    #     print(f"mutual_self_attention attn1 mask_fg.keys(): {mask_fg.keys()}", flush=True)
                    #     print(f"mutual_self_attention attn1 math.sqrt(norm_hidden_states.shape[1]): {math.sqrt(norm_hidden_states.shape[1])}", flush=True)
                    # if mask_fg is None or math.sqrt(norm_hidden_states.shape[1]) not in mask_fg or 'downblock_0_layer_1' not in block_name:
                    if mask_fg is None or math.sqrt(norm_hidden_states.shape[1]) not in mask_fg:
                        hidden_states_uc = (
                            self.attn1(
                                norm_hidden_states,
                                encoder_hidden_states=modify_norm_hidden_states,
                                attention_mask=attention_mask,
                                # save_attention=True,
                                # block_name='attn1',
                                # max_attention_size=512,  # Adjust this value based on your GPU memory
                                # spatial_attention=True,
                                # spatial_size=64,
                                # grid_size=32,
                                attention_name=block_name+'_attn1',
                            )
                            + hidden_states
                        )
                    
                    else:
                        print(f"mutual_self_attention attn1 mask_fg.keys(): {mask_fg.keys()}", flush=True)
                    # if face_masks is not None and math.sqrt(norm_hidden_states.shape[1]) in face_masks.keys():
                        print(f"mutual_self_attention attn1 fg mask size: {math.sqrt(norm_hidden_states.shape[1])}", flush=True)
                        # fg_mask = mask_fg[math.sqrt(norm_hidden_states.shape[1])].flatten() # 64x64 or 32x32
                        # bool_fg_mask = torch.from_numpy(fg_mask).bool()

                        # bool_fg_mask = mask_fg[math.sqrt(norm_hidden_states.shape[1])]["fg"]
                        # # bool_bg_mask = mask_fg[math.sqrt(norm_hidden_states.shape[1])]["bg"]

                        # masked_norm_hidden_states_fg = norm_hidden_states[:, bool_fg_mask, :]
                        # masked_norm_hidden_states_bg = norm_hidden_states[:, ~bool_fg_mask, :]
                        # # masked_norm_hidden_states_bg = norm_hidden_states[:, bool_bg_mask, :]
                        # bool_fg_mask_encoder = torch.cat([bool_fg_mask, bool_fg_mask], dim=0)
                        # masked_modify_norm_hidden_states_fg = modify_norm_hidden_states[:, bool_fg_mask_encoder, :]
                        # masked_modify_norm_hidden_states_bg = modify_norm_hidden_states[:, ~bool_fg_mask_encoder, :]

                        bool_fg_mask = mask_fg[math.sqrt(norm_hidden_states.shape[1])]["fg"]
                        bool_bg_mask = mask_fg[math.sqrt(norm_hidden_states.shape[1])]["bg"]
                        bool_fg_mask_encoder = mask_fg[math.sqrt(norm_hidden_states.shape[1])]["fg_encoder"]
                        bool_bg_mask_encoder = mask_fg[math.sqrt(norm_hidden_states.shape[1])]["bg_encoder"]

                        masked_norm_hidden_states_fg = norm_hidden_states[:, bool_fg_mask, :]
                        masked_norm_hidden_states_bg = norm_hidden_states[:, bool_bg_mask, :]
                        masked_modify_norm_hidden_states_fg = modify_norm_hidden_states[:, bool_fg_mask_encoder, :]
                        masked_modify_norm_hidden_states_bg = modify_norm_hidden_states[:, bool_bg_mask_encoder, :]


                        hidden_states_uc_fg = (
                            self.attn1(
                                masked_norm_hidden_states_fg,
                                encoder_hidden_states=masked_modify_norm_hidden_states_fg,
                                attention_mask=attention_mask,
                                attention_name=block_name+'_attn1_fg',
                            )
                        )

                        hidden_states_uc_bg = (
                            self.attn1(
                                masked_norm_hidden_states_bg,
                                encoder_hidden_states=masked_modify_norm_hidden_states_bg,
                                attention_mask=attention_mask,
                                attention_name=block_name+'_attn1_bg',
                            )
                        )

                        print(f"mutual_self_attention attn1 hidden_states_uc_fg shape: {hidden_states_uc_fg.shape}", flush=True)
                        print(f"mutual_self_attention attn1 hidden_states_uc_bg shape: {hidden_states_uc_bg.shape}", flush=True)

                        hidden_states_uc = torch.zeros_like(hidden_states)
                        hidden_states_uc[:, bool_fg_mask, :] = hidden_states_uc[:, bool_fg_mask, :] + hidden_states_uc_fg
                        hidden_states_uc[:, ~bool_fg_mask, :] = hidden_states_uc[:, ~bool_fg_mask, :] + hidden_states_uc_bg

                        hidden_states_uc = hidden_states_uc + hidden_states
                        

                    # else:
                    #     hidden_states_uc = (
                    #         self.attn1(
                    #             norm_hidden_states,
                    #             encoder_hidden_states=modify_norm_hidden_states,
                    #             attention_mask=attention_mask,
                    #             # save_attention=True,
                    #             # block_name='attn1',
                    #             # max_attention_size=512,  # Adjust this value based on your GPU memory
                    #             # spatial_attention=True,
                    #             # spatial_size=64,
                    #             # grid_size=32,
                    #             attention_name=block_name+'_attn1',
                    #         )
                    #         + hidden_states
                    #     )
                    
                    end_event.record()
                    end_event.synchronize()
                    time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
                    print(f"mutual_self_attention attn1 took {time_elapsed} seconds", flush=True)
                    # print(f"mutual_self_attention attn1 scores shape: {attn1_scores.shape}", flush=True)
            
                    # assert 0

                    if do_classifier_free_guidance:
                        # start_event.record()
                        hidden_states_c = hidden_states_uc.clone()
                        _uc_mask = uc_mask.clone()
                        if hidden_states.shape[0] != _uc_mask.shape[0]:
                            _uc_mask = (
                                torch.Tensor(
                                    [1] * (hidden_states.shape[0] // 2)
                                    + [0] * (hidden_states.shape[0] // 2)
                                )
                                .to(device)
                                .bool()
                            )
                        # print(f"mutual_self_attention before cfg attn1 took {time.time() - start_time} seconds", flush=True)
                        start_event.record()
                        # masked_norm_hidden_states = norm_hidden_states[_uc_mask]
                        # masked_hidden_states = hidden_states[_uc_mask]
                        # print(f"mutual_self_attention mask hidden states took {time.time() - start_time} seconds", flush=True)
                        # start_event.record()
                        # temp = (
                        #     self.attn1(
                        #         masked_norm_hidden_states,
                        #         encoder_hidden_states=masked_norm_hidden_states,
                        #         attention_mask=attention_mask,
                        #     )
                        #     + masked_hidden_states
                        # )
                        # temp = (
                        #     self.attn1(
                        #         norm_hidden_states[_uc_mask],
                        #         encoder_hidden_states=norm_hidden_states[_uc_mask],
                        #         attention_mask=attention_mask,
                        #     )
                        #     + hidden_states[_uc_mask]
                        # )
                        # print(f"mutual_self_attention cfg attn1 computation took {time.time() - start_time} seconds", flush=True)
                        # start_event.record()
                        # hidden_states_c[_uc_mask] = temp
                        # print(f"mutual_self_attention cfg attn1 assignment took {time.time() - start_time} seconds", flush=True)

                        hidden_states_c[_uc_mask] = (
                            self.attn1(
                                norm_hidden_states[_uc_mask],
                                encoder_hidden_states=norm_hidden_states[_uc_mask],
                                attention_mask=attention_mask,
                                attention_name=block_name+'_cfg_attn1',
                            )
                            + hidden_states[_uc_mask]
                        )
                        end_event.record()
                        end_event.synchronize()
                        time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
                        print(f"mutual_self_attention cfg attn1 took {time_elapsed} seconds", flush=True)
                        hidden_states = hidden_states_c.clone()
                    else:
                        hidden_states = hidden_states_uc

                    # self.bank.clear()
                    if self.attn2 is not None:
                        # Cross-Attention
                        start_event.record()
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )
                        end_event.record()
                        end_event.synchronize()
                        time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
                        print(f"mutual_self_attention norm2 took {time_elapsed} seconds", flush=True)
                        start_event.record()
                        # print(f"mutual_self_attention attn2 norm_hidden_states shape: {norm_hidden_states.shape}", flush=True)
                        # print(f"mutual_self_attention attn2 encoder_hidden_states shape: {encoder_hidden_states.shape}", flush=True)
                        hidden_states = (
                            self.attn2(
                                norm_hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                attention_mask=attention_mask,
                                attention_name=block_name+'_attn2',
                            )
                            + hidden_states
                        )
                        end_event.record()
                        end_event.synchronize()
                        time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
                        print(f"mutual_self_attention attn2 took {time_elapsed} seconds", flush=True)
                    # Feed-forward
                    start_event.record()
                    hidden_states = self.ff(self.norm3(
                        hidden_states)) + hidden_states
                    end_event.record()
                    end_event.synchronize()
                    time_elapsed = start_event.elapsed_time(end_event) / 1000  # ms to s
                    print(f"mutual_self_attention ff took {time_elapsed} seconds", flush=True)

                    # Temporal-Attention
                    if self.unet_use_temporal_attention:
                        d = hidden_states.shape[1]
                        hidden_states = rearrange(
                            hidden_states, "(b f) d c -> (b d) f c", f=video_length
                        )
                        norm_hidden_states = (
                            self.norm_temp(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm_temp(hidden_states)
                        )
                        hidden_states = (
                            self.attn_temp(norm_hidden_states) + hidden_states
                        )
                        hidden_states = rearrange(
                            hidden_states, "(b d) f c -> (b f) d c", d=d
                        )

                    return hidden_states, motion_frames_fea

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                tmp = norm_hidden_states.shape[0] // encoder_hidden_states.shape[0]
                attn_output = self.attn2(
                    norm_hidden_states,
                    # TODO: repeat这个地方需要斟酌一下
                    encoder_hidden_states=encoder_hidden_states.repeat(
                        tmp, 1, 1),
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                    norm_hidden_states *
                    (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) +
                        torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, (BasicTransformerBlock, TemporalBasicTransformerBlock))
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, (BasicTransformerBlock, TemporalBasicTransformerBlock))
                ]
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, BasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module,
                        BasicTransformerBlock)
                if isinstance(module, TemporalBasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module,
                        TemporalBasicTransformerBlock)

                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

    def update(self, writer, dtype=torch.float16):
        """
        Update the model's parameters.

        Args:
            writer (torch.nn.Module): The model's writer object.
            dtype (torch.dtype, optional): The data type to be used for the update. Defaults to torch.float16.

        Returns:
            None.
        """
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) +
                        torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in (
                        torch_dfs(writer.unet.mid_block)
                        + torch_dfs(writer.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in torch_dfs(writer.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]

            assert len(reader_attn_modules) == len(writer_attn_modules)
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            writer_attn_modules = sorted(
                writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                r.bank = [v.clone().to(dtype) for v in w.bank]


    def clear(self):
        """
        Clears the attention bank of all reader attention modules.

        This method is used when the `reference_attn` attribute is set to `True`.
        It clears the attention bank of all reader attention modules inside the UNet
        model based on the selected `fusion_blocks` mode.

        If `fusion_blocks` is set to "midup", it searches for reader attention modules
        in both the mid block and up blocks of the UNet model. If `fusion_blocks` is set
        to "full", it searches for reader attention modules in the entire UNet model.

        It sorts the reader attention modules by the number of neurons in their
        `norm1.normalized_shape[0]` attribute in descending order. This sorting ensures
        that the modules with more neurons are cleared first.

        Finally, it iterates through the sorted list of reader attention modules and
        calls the `clear()` method on each module's `bank` attribute to clear the
        attention bank.
        """
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) +
                        torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, (BasicTransformerBlock, TemporalBasicTransformerBlock))
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, (BasicTransformerBlock, TemporalBasicTransformerBlock))
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r in reader_attn_modules:
                r.bank.clear()