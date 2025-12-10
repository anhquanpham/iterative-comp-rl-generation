import dataclasses
import math

from ...base import DeviceArg, LearnableConfig, register_learnable
from ...constants import ActionSpace
from ...models.builders import (
    create_categorical_policy,
    create_continuous_q_function,
    create_normal_policy,
    create_parameter,
)
from ...models.encoders import EncoderFactory, make_encoder_field
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...optimizers.optimizers import OptimizerFactory, make_optimizer_field
from ...types import Shape
from .base import QLearningAlgoBase

from typing import Optional, Sequence, Dict, Any
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory, VectorEncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.models.torch.encoders import Encoder, EncoderWithAction, _VectorEncoder
from d3rlpy.algos.torch.ddpg_impl import DDPGBaseImpl
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler

from d3rlpy.models.torch.policies import NormalPolicy
from d3rlpy.models.torch.q_functions.mean_q_function import ContinuousMeanQFunction
from d3rlpy.models.torch import (
    ValueFunction,
)

from d3rlpy.models.builders import (
    create_non_squashed_normal_policy,
)

import dataclasses
import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.optim import Optimizer

from ....models.torch import (
    ActionOutput,
    CategoricalPolicy,
    ContinuousEnsembleQFunctionForwarder,
    NormalPolicy,
    Parameter,
    Policy,
    build_squashed_gaussian_distribution,
    get_parameter,
)
from ....optimizers import OptimizerWrapper
from ....torch_utility import (
    CudaGraphWrapper,
    Modules,
    TorchMiniBatch,
    hard_sync,
)
from ....types import Shape, TorchObservation
from ..base import QLearningAlgoImplBase
from .ddpg_impl import DDPGBaseActorLoss, DDPGBaseImpl, DDPGBaseModules



__all__ = ["SACConfig", "SAC"]


@dataclasses.dataclass()
class SACConfig(LearnableConfig):
    r"""Config Soft Actor-Critic algorithm.

    SAC is a DDPG-based maximum entropy RL algorithm, which produces
    state-of-the-art performance in online RL settings.
    SAC leverages twin Q functions proposed in TD3. Additionally,
    `delayed policy update` in TD3 is also implemented, which is not done in
    the paper.

    .. math::

        L(\theta_i) = \mathbb{E}_{s_t,\, a_t,\, r_{t+1},\, s_{t+1} \sim D,\,
                                   a_{t+1} \sim \pi_\phi(\cdot|s_{t+1})} \Big[
            \big(y - Q_{\theta_i}(s_t, a_t)\big)^2\Big]

    .. math::

        y = r_{t+1} + \gamma \Big(\min_j Q_{\theta_j}(s_{t+1}, a_{t+1})
            - \alpha \log \big(\pi_\phi(a_{t+1}|s_{t+1})\big)\Big)

    .. math::

        J(\phi) = \mathbb{E}_{s_t \sim D,\, a_t \sim \pi_\phi(\cdot|s_t)}
            \Big[\alpha \log (\pi_\phi (a_t|s_t))
              - \min_i Q_{\theta_i}\big(s_t, \pi_\phi(a_t|s_t)\big)\Big]

    The temperature parameter :math:`\alpha` is also automatically adjustable.

    .. math::

        J(\alpha) = \mathbb{E}_{s_t \sim D,\, a_t \sim \pi_\phi(\cdot|s_t)}
            \bigg[-\alpha \Big(\log \big(\pi_\phi(a_t|s_t)\big) + H\Big)\bigg]

    where :math:`H` is a target
    entropy, which is defined as :math:`\dim a`.

    References:
        * `Haarnoja et al., Soft Actor-Critic: Off-Policy Maximum Entropy Deep
          Reinforcement Learning with a Stochastic Actor.
          <https://arxiv.org/abs/1801.01290>`_
        * `Haarnoja et al., Soft Actor-Critic Algorithms and Applications.
          <https://arxiv.org/abs/1812.05905>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        actor_learning_rate (float): Learning rate for policy function.
        critic_learning_rate (float): Learning rate for Q functions.
        temp_learning_rate (float): Learning rate for temperature parameter.
        actor_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the actor.
        critic_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the critic.
        temp_optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory for the temperature.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory for the critic.
        q_func_factory (d3rlpy.models.q_functions.QFunctionFactory):
            Q function factory.
        batch_size (int): Mini-batch size.
        gamma (float): Discount factor.
        tau (float): Target network synchronization coefficiency.
        n_critics (int): Number of Q functions for ensemble.
        initial_temperature (float): Initial temperature value.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temp_learning_rate: float = 3e-4
    actor_optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    temp_optim_factory: OptimizerFactory = make_optimizer_field()
    actor_encoder_factory: EncoderFactory = dataclasses.field(
        default_factory=lambda: create_cp_encoderfactory()
    )
    critic_encoder_factory: EncoderFactory = dataclasses.field(
        default_factory=lambda: create_cp_encoderfactory(with_action=True)
    )
    q_func_factory: Any = dataclasses.field(
        default_factory=CompositionalMeanQFunctionFactory
    )
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    n_critics: int = 2
    initial_temperature: float = 1.0

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "SAC":
        return SAC(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "cp_sac"


class SAC(QLearningAlgoBase[SACImpl, SACConfig]):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        # compositional actor
        policy = create_non_squashed_normal_policy(
            observation_shape,
            action_size,
            self._config.actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
        )
        # compositional critics
        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        log_temp = create_parameter(
            (1, 1),
            math.log(self._config.initial_temperature),
            device=self._device,
            enable_ddp=self._enable_ddp,
        )

        actor_optim = self._config.actor_optim_factory.create(
            policy.named_modules(),
            lr=self._config.actor_learning_rate,
            compiled=self.compiled,
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.named_modules(),
            lr=self._config.critic_learning_rate,
            compiled=self.compiled,
        )
        if self._config.temp_learning_rate > 0:
            temp_optim = self._config.temp_optim_factory.create(
                log_temp.named_modules(),
                lr=self._config.temp_learning_rate,
                compiled=self.compiled,
            )
        else:
            temp_optim = None

        modules = SACModules(
            policy=policy,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            log_temp=log_temp,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            temp_optim=temp_optim,
        )

        self._impl = SACImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=self._config.gamma,
            tau=self._config.tau,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS

register_learnable(SACConfig)




__all__ = [
    "SACImpl",
    "SACModules",
    "SACActorLoss",
]


@dataclasses.dataclass(frozen=True)
class SACModules(DDPGBaseModules):
    policy: NormalPolicy
    log_temp: Parameter
    temp_optim: Optional[OptimizerWrapper]


@dataclasses.dataclass(frozen=True)
class SACActorLoss(DDPGBaseActorLoss):
    temp: torch.Tensor
    temp_loss: torch.Tensor


class SACImpl(DDPGBaseImpl):
    _modules: SACModules

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: SACModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        compiled: bool,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
            tau=tau,
            compiled=compiled,
            device=device,
        )

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> SACActorLoss:
        dist = build_squashed_gaussian_distribution(action)
        sampled_action, log_prob = dist.sample_with_log_prob()

        if self._modules.temp_optim:
            temp_loss = self.update_temp(log_prob)
        else:
            temp_loss = torch.tensor(
                0.0, dtype=torch.float32, device=sampled_action.device
            )

        entropy = get_parameter(self._modules.log_temp).exp() * log_prob
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, sampled_action, "min"
        )
        return SACActorLoss(
            actor_loss=(entropy - q_t).mean(),
            temp_loss=temp_loss,
            temp=get_parameter(self._modules.log_temp).exp()[0][0],
        )

    def update_temp(self, log_prob: torch.Tensor) -> torch.Tensor:
        assert self._modules.temp_optim
        self._modules.temp_optim.zero_grad()
        with torch.no_grad():
            targ_temp = log_prob - self._action_size
        loss = -(get_parameter(self._modules.log_temp).exp() * targ_temp).mean()
        loss.backward()
        self._modules.temp_optim.step()
        return loss

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            dist = build_squashed_gaussian_distribution(
                self._modules.policy(batch.next_observations)
            )
            action, log_prob = dist.sample_with_log_prob()
            entropy = get_parameter(self._modules.log_temp).exp() * log_prob
            target = self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
            return target - entropy

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        dist = build_squashed_gaussian_distribution(self._modules.policy(x))
        return dist.sample()





DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def fanin_init(tensor):
    """Initialize the weights of a layer with fan-in initialization.

    Args:
        tensor (torch.Tensor): Tensor to initialize.

    Returns:
        torch.Tensor: Initialized tensor.

    Raises:
        Exception: If the shape of the tensor is less than 2.
    """
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class CompositionalMlp(nn.Module):
    """Compositional MLP module."""

    def __init__(
        self,
        sizes: Sequence[Sequence[int]],
        num_modules: Sequence[int],
        module_assignment_positions: Sequence[int],
        module_inputs: Sequence[str],
        interface_depths: Sequence[int],
        graph_structure: Sequence[Sequence[int]],
        init_w: float = 3e-3,
        hidden_activation: nn.Module = nn.ReLU,
        output_activation: nn.Module = nn.Identity,
        hidden_init: Optional[nn.Module] = fanin_init,
        b_init_value: float = 0.1,
        layer_norm: bool = False,
        layer_norm_kwargs: Optional[dict] = None,
    ):
        """Initialize the compositional MLP module.

        Args:
            sizes (list): List of sizes of each layer.
            num_modules (list): List of number of modules of each type.
            module_assignment_positions (list): List of module assignment positions.
            module_inputs (list): List of module inputs.
            interface_depths (list): List of interface depths.
            graph_structure (list): List of graph structures.
            init_w (float, optional): Initial weight value. Defaults to 3e-3.
            hidden_activation (nn.Module, optional): Hidden activation module. Defaults to nn.ReLU.
            output_activation (nn.Module, optional): Output activation module. Defaults to nn.Identity.
            hidden_init (function, optional): Hidden initialization function. Defaults to fanin_init.
            b_init_value (float, optional): Initial bias value. Defaults to 0.1.
            layer_norm (bool, optional): Whether to use layer normalization. Defaults to False.
            layer_norm_kwargs (dict, optional): Keyword arguments for layer normalization. Defaults to None.
        """
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.sizes = sizes
        self.num_modules = num_modules
        self.module_assignment_positions = module_assignment_positions
        self.module_inputs = module_inputs  # keys in a dict
        self.interface_depths = interface_depths
        self.graph_structure = (
            graph_structure  # [[0], [1,2], 3] or [[0], [1], [2], [3]]
        )
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        self.count = 0

        self.module_list = nn.ModuleList()  # e.g., object, robot, task...

        for graph_depth in range(
            len(graph_structure)
        ):  # root -> children -> ... leaves
            for j in graph_structure[
                graph_depth
            ]:  # loop over all module types at this depth
                self.module_list.append(nn.ModuleDict())  # pre, post
                self.module_list[j]["pre_interface"] = nn.ModuleList()
                self.module_list[j]["post_interface"] = nn.ModuleList()
                for k in range(num_modules[j]):  # loop over all modules of this type
                    layers_pre = []
                    layers_post = []
                    for i in range(
                        len(sizes[j]) - 1
                    ):  # loop over all depths in this module
                        if i == interface_depths[j]:
                            input_size = sum(
                                sizes[j_prev][-1]
                                for j_prev in graph_structure[graph_depth - 1]
                            )
                            input_size += sizes[j][i]
                        else:
                            input_size = sizes[j][i]

                        fc = nn.Linear(input_size, sizes[j][i + 1])
                        if (
                            graph_depth < len(graph_structure) - 1
                            or i < len(sizes[j]) - 2
                        ):
                            hidden_init(fc.weight)
                            fc.bias.data.fill_(b_init_value)
                            act = hidden_activation
                            layer_norm_this = layer_norm
                        else:
                            fc.weight.data.uniform_(-init_w, init_w)
                            fc.bias.data.uniform_(-init_w, init_w)
                            act = output_activation
                            layer_norm_this = None

                        if layer_norm_this is not None:
                            new_layer = [fc, nn.LayerNorm(sizes[j][i + 1]), act()]
                        else:
                            new_layer = [fc, act()]

                        if i < interface_depths[j]:
                            layers_pre += new_layer
                        else:
                            layers_post += new_layer
                    if layers_pre:
                        self.module_list[j]["pre_interface"].append(
                            nn.Sequential(*layers_pre)
                        )
                    else:  # it's either a root or a module with no preprocessing
                        self.module_list[j]["pre_interface"].append(nn.Identity())
                    self.module_list[j]["post_interface"].append(
                        nn.Sequential(*layers_post)
                    )

    def forward(self, input_val: torch.Tensor, return_preactivations: bool = False):
        """Forward pass.

        Args:
            input_val (torch.Tensor): Input tensor.
            return_preactivations (bool, optional): Whether to return preactivations. Defaults to False.

        Returns:
            torch.Tensor: Output tensor.
        """
        if len(input_val.shape) > 2:
            input_val = input_val.squeeze(0)

        if return_preactivations:
            raise NotImplementedError("TODO: implement return preactivations")
        x = None
        for graph_depth in range(
            len(self.graph_structure)
        ):  # root -> children -> ... -> leaves
            x_post = []  # in case multiple module types at the same depth in the graph
            for j in self.graph_structure[graph_depth]:  # nodes (modules) at this depth
                if len(input_val.shape) == 1:
                    x_pre = input_val[self.module_inputs[j]]
                    onehot = input_val[self.module_assignment_positions[j]]
                    module_index = onehot.nonzero()[0]
                    x_pre = self.module_list[j]["pre_interface"][module_index](x_pre)
                    if x is not None:
                        x_pre = torch.cat((x, x_pre), dim=-1)
                    x_post.append(
                        self.module_list[j]["post_interface"][module_index](x_pre)
                    )
                else:
                    x_post_tmp = torch.empty(input_val.shape[0], self.sizes[j][-1]).to(
                        DEVICE
                    )
                    x_pre = input_val[:, self.module_inputs[j]]
                    onehot = input_val[:, self.module_assignment_positions[j]]
                    module_indices = onehot.nonzero(as_tuple=True)
                    assert (
                        module_indices[0]
                        == torch.arange(module_indices[0].shape[0]).to(DEVICE)
                    ).all()
                    module_indices_1 = module_indices[1]
                    for module_idx in range(self.num_modules[j]):
                        mask_inputs_for_this_module = module_indices_1 == module_idx
                        mask_to_input_idx = mask_inputs_for_this_module.nonzero()
                        x_pre_this_module = self.module_list[j]["pre_interface"][
                            module_idx
                        ](x_pre[mask_inputs_for_this_module])
                        if x is not None:
                            x_pre_this_module = torch.cat(
                                (x[mask_inputs_for_this_module], x_pre_this_module),
                                dim=-1,
                            )
                        x_post_this_module = self.module_list[j]["post_interface"][
                            module_idx
                        ](x_pre_this_module)
                        mask_to_input_idx = mask_to_input_idx.expand(
                            mask_to_input_idx.shape[0], x_post_this_module.shape[1]
                        )
                        x_post_tmp.scatter_(0, mask_to_input_idx, x_post_this_module)
                    x_post.append(x_post_tmp)
            x = torch.cat(x_post, dim=-1)
        return x


class _CompositionalEncoder(_VectorEncoder):  # type: ignore
    """_CompositionalEncoder class for d3rlpy."""

    def __init__(
        self,
        encoder_kwargs: dict,
        observation_shape: Sequence[int],
        init_w: float = 3e-3,
        *args,
        **kwargs,
    ):
        """Initialize _CompositionalEncoder class.

        Args:
            encoder_kwargs (dict): Encoder parameters.
            observation_shape (Sequence[int]): Observation shape.
            init_w (float, optional): Initial weight. Defaults to 3e-3.
        """
        super().__init__(
            observation_shape,
            hidden_units=None,
            use_batch_norm=False,
            dropout_rate=None,
            use_dense=False,
            activation=nn.ReLU(),
        )

        self._observation_shape = observation_shape
        self.encoder_kwargs = encoder_kwargs
        sizes = encoder_kwargs["sizes"]
        output_dim = encoder_kwargs["output_dim"]
        num_modules = encoder_kwargs["num_modules"]
        module_assignment_positions = encoder_kwargs["module_assignment_positions"]
        module_inputs = encoder_kwargs["module_inputs"]
        interface_depths = encoder_kwargs["interface_depths"]
        graph_structure = encoder_kwargs["graph_structure"]
        sizes = list(sizes)
        for j in range(len(sizes)):
            input_size = len(module_inputs[j])
            sizes[j] = [input_size] + list(sizes[j])
            if j in graph_structure[-1]:
                sizes[j] = sizes[j] + [output_dim]

        self._feature_size = sizes[-1][-1]

        self.comp_mlp = CompositionalMlp(
            sizes=sizes,
            num_modules=num_modules,
            module_assignment_positions=module_assignment_positions,
            module_inputs=module_inputs,
            interface_depths=interface_depths,
            graph_structure=graph_structure,
            init_w=init_w,
        )

    def _fc_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.comp_mlp.forward(x)

    @property
    def last_layer(self) -> nn.Linear:
        raise NotImplementedError("CompositionalEncoder does not have last_layer")


class CompositionalEncoder(_CompositionalEncoder, Encoder):
    """Implements the actual Compositional Encoder."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simply runs the forward pass from _CompositionalEncoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self._fc_encode(x)


class CompositionalEncoderWithAction(_CompositionalEncoder, EncoderWithAction):
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, action], dim=1)
        h = self._fc_encode(x)
        return h


class CompositionalNonSquashedNormalPolicy(NormalPolicy):
    """CompositionalNonSquashedNormalPolicy class for d3rlpy."""

    def __init__(self, *args, **kwargs):
        """Initialize CompositionalNonSquashedNormalPolicy."""
        super().__init__(
            squash_distribution=False,
            *args,
            **kwargs,
        )
        self._mu = nn.Identity()


def create_non_squashed_normal_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
    use_std_parameter: bool = False,
) -> CompositionalNonSquashedNormalPolicy:
    """Create a non-squashed normal policy.

    Args:
        observation_shape (Sequence[int]): Observation shape.
        action_size (int): Action size.
        encoder_factory (EncoderFactory): Encoder factory.
        min_logstd (float, optional): Minimum log standard deviation.
            Defaults to -20.0.
        max_logstd (float, optional): Maximum log standard deviation.
            Defaults to 2.0.
        use_std_parameter (bool, optional): Use std parameter. Defaults to False.

    Returns:
        CompositionalNonSquashedNormalPolicy: Non-squashed normal policy.
    """
    encoder = encoder_factory.create(observation_shape)
    return CompositionalNonSquashedNormalPolicy(
        encoder,
        action_size,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
        use_std_parameter=use_std_parameter,
    )

def create_compositional_value_function(
    observation_shape: Sequence[int], encoder_factory: EncoderFactory
) -> ValueFunction:
    encoder = encoder_factory.create(observation_shape)
    return CompositionalValueFunction(encoder)


class CompositionalValueFunction(ValueFunction):
    def __init__(self, encoder: Encoder):
        super().__init__(encoder)
        self._fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._encoder(x)


class CompositionalContinuousMeanQFunction(ContinuousMeanQFunction):
    def __init__(self, encoder: EncoderWithAction):
        super().__init__(encoder)
        self._fc = nn.Identity()

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, action)


class CompositionalMeanQFunctionFactory(MeanQFunctionFactory):
    def create_discrete(
        self,
        encoder: Encoder,
        action_size: int,
    ):
        raise NotImplementedError(
            "CompositionalMeanQFunctionFactory does not support discrete action spaces"
        )

    def create_continuous(
        self,
        encoder: EncoderWithAction,
    ) -> CompositionalContinuousMeanQFunction:
        return CompositionalContinuousMeanQFunction(encoder)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "share_encoder": self._share_encoder,
        }


class CompositionalEncoderFactory(VectorEncoderFactory):
    """Encoder factory for CompositionalEncoder."""

    def __init__(self, encoder_kwargs: dict, *args, **kwargs):
        """Initialize CompositionalEncoderFactory."""
        super().__init__(*args, **kwargs)
        self.encoder_kwargs = encoder_kwargs

    def create(self, observation_shape: Sequence[int]) -> CompositionalEncoder:
        """Create a CompositionalEncoder."""
        assert len(observation_shape) == 1
        return CompositionalEncoder(
            encoder_kwargs=self.encoder_kwargs,
            observation_shape=observation_shape,
        )

    def create_with_action(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        discrete_action: bool = False,
    ) -> CompositionalEncoderWithAction:
        return CompositionalEncoderWithAction(
            encoder_kwargs=self.encoder_kwargs,
            observation_shape=observation_shape,
            action_size=action_size,
            discrete_action=discrete_action,
        )


def create_cp_encoderfactory(with_action=False, output_dim=None):
    obs_dim = 93
    act_dim = 8
    # fmt: off
    observation_positions = {
        'object-state': np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13]), 
        'obstacle-state': np.array([14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]), 
        'goal-state': np.array([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]), 
        'object_id': np.array([45, 46, 47, 48]), 
        'robot_id': np.array([49, 50, 51, 52]), 
        'obstacle_id': np.array([53, 54, 55, 56]), 
        'subtask_id': np.array([57, 58, 59, 60]), 
        'robot0_proprio-state': np.array([61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
            78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92])}
    if with_action:
        observation_positions["action"] = np.array([93, 94, 95, 96, 97, 98, 99, 100])
    # fmt: on

    sizes = ((32,), (32, 32), (64, 64, 64), (64, 64, 64))
    module_names = ["obstacle_id", "object_id", "subtask_id", "robot_id"]
    module_input_names = [
        "obstacle-state",
        "object-state",
        "goal-state",
    ]
    if with_action:
        module_input_names.append(["robot0_proprio-state", "action"])
    else:
        module_input_names.append("robot0_proprio-state")

    module_assignment_positions = [observation_positions[key] for key in module_names]
    interface_depths = [-1, 1, 2, 3]
    graph_structure = [[0], [1], [2], [3]]
    num_modules = [len(onehot_pos) for onehot_pos in module_assignment_positions]

    module_inputs = []
    for key in module_input_names:
        if isinstance(key, list):
            # concatenate the inputs
            module_inputs.append(
                np.concatenate([observation_positions[k] for k in key], axis=0)
            )
        else:
            module_inputs.append(observation_positions[key])

    # module_inputs = [observation_positions[key]  for key in module_input_names]

    encoder_kwargs = {
        "sizes": sizes,
        "obs_dim": obs_dim,
        "output_dim": output_dim if output_dim is not None else act_dim,
        "num_modules": num_modules,
        "module_assignment_positions": module_assignment_positions,
        "module_inputs": module_inputs,
        "interface_depths": interface_depths,
        "graph_structure": graph_structure,
    }

    fac = CompositionalEncoderFactory(
        encoder_kwargs,
    )

    return fac