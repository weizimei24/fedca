"""Algorithm capabilities and complete state, independent of datasets."""
from dataclasses import dataclass
import math

import torch

from utils.runtime_utils import cpu_tree


@dataclass(frozen=True)
class Method:
    frozen_a: bool = False
    changes_base: bool = False
    parameters: tuple = ()


METHODS = {
    'fedit': Method(), 'centralized': Method(), 'localonly': Method(),
    'ffalora': Method(frozen_a=True), 'fedsvd': Method(frozen_a=True),
    'fedrotlora': Method(parameters=('lam',)), 'flexlora': Method(),
    'fedexlora': Method(changes_base=True),
    'fedmomentum': Method(changes_base=True, parameters=('residual_threshold',)),
    'fedca': Method(parameters=('ca_lambda', 'ca_tau', 'ca_eta_mode', 'ca_eta',
        'ca_eta_grid', 'ca_kappa', 'ca_beta', 'ca_probe_size', 'ca_probe_bs', 'ca_probe_diag')),
}


def method_spec(name):
    if name not in METHODS:
        raise ValueError(f'Method {name!r} has no shared state adapter; register it in method_state.METHODS')
    return METHODS[name]


def validate_method(args):
    method_spec(args.alg)
    if args.alg == 'fedca':
        from utils.ca_utils import parse_grid
        if not math.isfinite(args.ca_lambda) or args.ca_lambda < 0 or not math.isfinite(args.ca_tau) or args.ca_tau <= 0:
            raise ValueError('FedCA requires finite ca_lambda>=0 and ca_tau>0')
        if args.ca_probe_size <= 0 or args.ca_probe_bs <= 0 or args.ca_kappa < 0 or math.isnan(float(args.ca_kappa)):
            raise ValueError('FedCA requires positive probe size/batch and nonnegative kappa')
        if not math.isfinite(args.ca_beta) or not math.isfinite(args.ca_eta) or args.ca_eta <= 0:
            raise ValueError('FedCA requires finite beta and positive eta')
        parse_grid(args.ca_eta_grid)
    if args.alg == 'fedrotlora' and not 0 <= args.lam <= 1:
        raise ValueError('lam must be in [0, 1]')
    if args.alg == 'fedmomentum' and not 0 < args.residual_threshold <= 1:
        raise ValueError('residual_threshold must be in (0, 1]')


def configure_trainable(server=None, *, model=None, method=None):
    """Apply the same parameter policy at server setup and client entry."""
    if server is not None:
        model, method = server.model, server.args.alg
    if model is None or method is None:
        raise ValueError('configure_trainable requires a server or model and method')
    spec = method_spec(method)
    for name, param in model.named_parameters():
        if spec.frozen_a:
            # Original FFA-LoRA/FedSVD policy: freeze everything except B.
            param.requires_grad_('lora_B' in name)
        elif 'lora_' in name:
            param.requires_grad_(True)


def complete_adapter(model):
    """Includes frozen A: trainability does not determine model identity."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if 'lora_' in k}


def restore_tensors(model, values):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for key, value in values.items():
            if key not in params or params[key].shape != value.shape:
                raise ValueError(f'Model state shape/key mismatch: {key}')
            params[key].copy_(value.to(device=params[key].device, dtype=params[key].dtype))


def export_state(server):
    spec = method_spec(server.args.alg)
    state = server.model.state_dict()
    base = {}
    if spec.changes_base:
        for name in server.global_lora:
            if 'lora_A' in name:
                key = name.replace('.lora_A.default.weight', '.base_layer.weight')
                if key not in state:
                    raise ValueError(f'Missing mutable base layer: {key}')
                base[key] = state[key]
    result = {'adapter': cpu_tree(server.global_lora), 'base': cpu_tree(base)}
    if server.args.alg == 'localonly':
        result['local'] = cpu_tree(server.client_loras)
    return result


def restore_state(server, saved):
    expected = set(server.global_lora)
    if set(saved['adapter']) != expected:
        raise ValueError('Complete algorithm adapter keys differ')
    spec = method_spec(server.args.alg)
    expected_base = {name.replace('.lora_A.default.weight', '.base_layer.weight')
                     for name in expected if 'lora_A' in name} if spec.changes_base else set()
    if set(saved['base']) != expected_base:
        raise ValueError('Mutable backbone state is missing or unexpected')
    restore_tensors(server.model, {**saved['base'], **saved['adapter']})
    params = dict(server.model.named_parameters())
    server.global_lora = {k: v.to(params[k].device).clone() for k, v in saved['adapter'].items()}
    if server.args.alg == 'localonly':
        if set(saved.get('local', {})) != {c.id for c in server.clients}:
            raise ValueError('Local-only checkpoint client IDs differ')
        server.client_loras = {int(cid): {k: v.to(params[k].device).clone() for k, v in values.items()}
                               for cid, values in saved['local'].items()}
    configure_trainable(server)


def client_extras(client):
    if client.args.alg == 'fedca':
        return {'report': cpu_tree(client.report), 'last_excess': client.trainer.last_excess}
    return {}


def restore_client_extras(client, saved):
    if client.args.alg == 'fedca':
        client.report = saved['report']
        client.trainer.last_excess = saved['last_excess']


def active_extras(client):
    return {'fedca_statistics': client.trainer.checkpoint_statistics()} if client.args.alg == 'fedca' else {}
