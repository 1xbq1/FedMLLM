
import torch
import torch.nn as nn
import deepspeed
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer import *
from transformers.integrations import is_deepspeed_zero3_enabled
from sklearn.metrics import roc_auc_score
import re
from geomloss import SamplesLoss
import torch.nn.functional as F

train_pred_list = []
train_truth_list = []
train_ax = [0 for _ in range(2)]

class CPMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if not self.args.use_lora:
            outputs = self.model(data = inputs, use_cache=False)
        else:
            with self.model._enable_peft_forward_hooks(**inputs):
                outputs = self.model.base_model(data = inputs, use_cache=False)

        if labels is not None:
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            logits = outputs.logits.view(-1,
                                         self.model.config.vocab_size).contiguous()

            labels = labels.view(-1).long().contiguous()
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name)
                                   for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(
                            model, inputs, return_outputs=True
                        )
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
#             self.accelerator.backward(loss, retain_graph=True)

        return loss.detach() / self.args.gradient_accumulation_steps

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:

            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

class CPMTrainerReg(CPMTrainer):
    def __init__(self, global_state, mu, **kwargs):
        super(CPMTrainerReg, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = mu

    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(CPMTrainerReg, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            #print(name)
            name = name.replace("modules_to_save.", "")     # TODO: May need changes. to accord with peft
            name = name.replace("module.", "")
            name = name.replace("default.", "")
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss

# CPMTrainerSign
class CPMTrainerSign(CPMTrainer):
    def __init__(self, sign_RA, sign_RB, num_train_samples, lam_sign, **kwargs):
        super(CPMTrainerSign, self).__init__(**kwargs)
        self.sign_RA = sign_RA
        self.sign_RB = sign_RB
        self.num_train_samples = num_train_samples
        self.lam_sign = lam_sign

    def compute_loss(self, model, inputs, return_outputs=False, current_step=None):

        return_values = super(CPMTrainerSign, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        if current_step is not None:
            each_steps = max(1, (self.num_train_samples // self.args.per_device_train_batch_size) // self.args.gradient_accumulation_steps)
            total_steps = each_steps * self.args.num_train_epochs
            alpha = get_alpha(current_step, total_steps)
            lambda_orthog = get_lambda_orthog(alpha)
        else:
            print("no current_step")

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            #print(name)
            name = name.replace("modules_to_save.", "")     # TODO: May need changes. to accord with peft
            name = name.replace("module.", "")
            name = name.replace("default.", "")
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                if 'lora_A' in name:
                    ATA = torch.mm(param.T, param)
                    I = torch.eye(ATA.size(0), device=param.device)
                    loss += lambda_orthog * torch.norm(ATA - I, p='fro') ** 2
#                 if ('lora_A' in name) and (self.sign_RA is not None):
#                     loss += torch.norm(soft_sign(param, alpha) - self.sign_RA) / (torch.norm(self.sign_RA) + 1e-6)
# #                     loss += self.lam_sign * torch.norm(torch.sign(param) - self.sign_RA)
                if ('lora_B' in name) and (self.sign_RB is not None):
                    loss += torch.norm(soft_sign(param, alpha) - self.sign_RB) / (torch.norm(self.sign_RB) + 1e-6)
#                     loss += self.lam_sign * torch.norm(torch.sign(param) - self.sign_RB)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        # 计算当前训练步骤
        current_step = self.state.global_step  # 或者可以通过其他方式计算，如 `self.state.epoch * len(train_dataloader) + step`

        # 计算损失并返回
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, current_step=current_step)

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps


def soft_sign(x, alpha=10):
    return torch.tanh(alpha * x).cuda()

class CPMTrainerOTReg(CPMTrainer):
    def __init__(self, global_state, ot_w, **kwargs):
        super(CPMTrainerOTReg, self).__init__(**kwargs)
        self.global_state = global_state
        self.lambda_ot = ot_w
        self.q_proj_outputs = []
        self.v_proj_outputs = []

    def hook_q_proj(self, module, input, output):
        self.q_proj_outputs.append(output)

    def hook_v_proj(self, module, input, output):
        self.v_proj_outputs.append(output)

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if ("q_proj" in name) and (("lora_A" in name) or ("lora_B" in name)):
                module.register_forward_hook(self.hook_q_proj)
            elif ("v_proj" in name) and (("lora_A" in name) or ("lora_B" in name)):
                module.register_forward_hook(self.hook_v_proj)

    def compute_loss(self, model, inputs, return_outputs=False):
        self.register_hooks()
        return_values = super(CPMTrainerOTReg, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        new_q_features = []
        new_v_features = []
        target_length = -1
        for q_features, v_features in zip(self.q_proj_outputs, self.v_proj_outputs):
            target_length = max(max(q_features.shape[1], v_features.shape[1]), target_length)
        for q_features, v_features in zip(self.q_proj_outputs, self.v_proj_outputs):
            q_inter = interpolate_tensor(q_features, target_length)
            v_inter = interpolate_tensor(v_features, target_length)
            new_q_features.append(q_inter.squeeze(0))
            new_v_features.append(v_inter.squeeze(0))

        q_features_all = torch.cat(new_q_features, dim=-1)
        v_features_all = torch.cat(new_v_features, dim=-1)

        with torch.no_grad():
            U_q, _, _ = torch.svd_lowrank(q_features_all.cpu(), q=128)
            U_v, _, _ = torch.svd_lowrank(v_features_all.cpu(), q=128)

        U_q = U_q.to("cuda").clone().detach().requires_grad_(True)
        U_v = U_v.to("cuda").clone().detach().requires_grad_(True)

        ot_loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.1, scaling=0.8)

        loss_ot = ot_loss_fn(U_q, U_v)
        loss += self.lambda_ot * loss_ot

        return (loss, outputs) if return_outputs else loss

def get_alpha(current_step, total_steps, alpha_init=1, alpha_final=200):
    return alpha_final - 0.5 * (alpha_final - alpha_init) * (1 + np.cos(np.pi * current_step / total_steps))

def get_lambda_orthog(alpha_b, alpha_min=1, alpha_max=200, lambda_min=0.01, lambda_max=0.1):
    normalized = (alpha_b - alpha_min) / (alpha_max - alpha_min)
    return lambda_max - normalized * (lambda_max - lambda_min)

def interpolate_tensor(data, target_length):

    data = data.permute(0, 2, 1)
    interpolated_data = F.interpolate(data, size=target_length, mode='linear', align_corners=False)
    interpolated_data = interpolated_data.permute(0, 2, 1)

    return interpolated_data