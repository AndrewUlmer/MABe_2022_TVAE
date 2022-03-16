import torch
import torch.nn as nn
import numpy as np

from util.logging import LogEntry
from lib.distributions import Normal


class TVAE(nn.Module):
    """
    This is a modified version of the BaseSequentialModel class created by 
    Jennifer Sun located at https://github.com/neuroethology/TREBA/

    Modifications made are to include only the Trajectory Variational Autoencoder
    model i.e. there is no decoding into programs, just the original trajectory.
    """
    model_args = []
    requires_labels = False
    is_recurrent = True

    def __init__(self, model_config):
        super().__init__()

        if 'recurrent' in model_config:
            self.is_recurrent = model_config['recurrent']

        # Assert rnn_dim and num_layers are defined if model is recurrent
        if self.is_recurrent:
            if 'rnn_dim' not in self.model_args:
                self.model_args.append('rnn_dim') 
            if 'num_layers' not in self.model_args:
                self.model_args.append('num_layers')

        # Assert label_dim is defined if model requires labels
        if self.requires_labels and 'label_dim' not in self.model_args:
            self.model_args.append('label_dim')

        # Check for missing arguments
        missing_args = set(self.model_args) - set(model_config)
        assert len(missing_args) == 0, 'model_config is missing these arguments:\n\t{}'.format(', '.join(missing_args))
        
        self.config = model_config
        self.log = LogEntry()
        self.stage = 0 # some models have multi-stage training            

        self._construct_model()
        self._define_losses()

        if self.is_recurrent:
            assert hasattr(self, 'dec_rnn')

    def _construct_model(self):
        state_dim = self.config['state_dim']
        action_dim = self.config['action_dim']
        z_dim = self.config['z_dim']
        h_dim = self.config['h_dim']
        enc_rnn_dim = self.config['rnn_dim']
        dec_rnn_dim = self.config['rnn_dim'] if self.is_recurrent else 0
        label_rnn_dim = self.config['rnn_dim']
        num_layers = self.config['num_layers']

        # Define models used in TREBA.
        self.enc_birnn = nn.GRU(state_dim+action_dim, enc_rnn_dim, 
            num_layers=num_layers, bidirectional=True)

        label_dim = 0

        # Define TVAE Encoder and Decoder.
        self.enc_fc = nn.Sequential(
            nn.Linear(2*enc_rnn_dim+label_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_logvar = nn.Linear(h_dim, z_dim)

        self.dec_action_fc = nn.Sequential(
            nn.Linear(state_dim+z_dim+dec_rnn_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_action_mean = nn.Linear(h_dim, action_dim)
        self.dec_action_logvar = nn.Linear(h_dim, action_dim)

        # Whether the trajectory decoder is recurrent.
        if self.is_recurrent:
            self.dec_rnn = nn.GRU(state_dim+action_dim, dec_rnn_dim, num_layers=num_layers)

    def _define_losses(self):
        """
        losses must be added to log - if they are not, they will not 
        be optimized
        """
        self.log.add_loss('kl_div')
        self.log.add_loss('nll')
        self.log.add_metric('kl_div_true')

    @property
    def num_parameters(self):
        if not hasattr(self, '_num_parameters'):
            self._num_parameters = 0
            for p in self.parameters():
                count = 1
                for s in p.size():
                    count *= s
                self._num_parameters += count

        return self._num_parameters

    def init_hidden_state(self, batch_size=1):
        return torch.zeros(self.config['num_layers'], batch_size, self.config['rnn_dim'])

    def update_hidden(self, state, action):
        assert self.is_recurrent
        state_action_pair = torch.cat([state, action], dim=1).unsqueeze(0)
        hiddens, self.hidden = self.dec_rnn(state_action_pair, self.hidden)

        return hiddens

    def encode(self, states, actions=None, labels=None):
        enc_birnn_input = states
        if actions is not None:
            assert states.size(0) == actions.size(0)
            enc_birnn_input = torch.cat([states, actions], dim=-1)
        
        # Average is taken across time series passed into encoder
        hiddens, _ = self.enc_birnn(enc_birnn_input)
        avg_hiddens = torch.mean(hiddens, dim=0)
    
        # Mapping from averaged over time output and mean / std
        enc_fc_input = avg_hiddens
        if labels is not None:
            enc_fc_input = torch.cat([avg_hiddens, labels], 1)

        enc_h = self.enc_fc(enc_fc_input) if hasattr(self, 'enc_fc') else enc_fc_input
        enc_mean = self.enc_mean(enc_h)
        enc_logvar = self.enc_logvar(enc_h)

        return Normal(enc_mean, enc_logvar)

    def encode_mean(self, states, actions=None, labels=None):
        enc_birnn_input = states
        if actions is not None:
            assert states.size(0) == actions.size(0)
            enc_birnn_input = torch.cat([states, actions], dim=-1)
        
        hiddens, _ = self.enc_birnn(enc_birnn_input)
        avg_hiddens = torch.mean(hiddens, dim=0)

        enc_fc_input = avg_hiddens
        if labels is not None:
            enc_fc_input = torch.cat([avg_hiddens, labels], 1)

        enc_h = self.enc_fc(enc_fc_input) if hasattr(self, 'enc_fc') else enc_fc_input
        enc_mean = self.enc_mean(enc_h)
        enc_logvar = self.enc_logvar(enc_h)

        return enc_mean, hiddens     

    def decode_action(self, state):
        dec_fc_input = torch.cat([state, self.z], dim=1)

        if self.is_recurrent:
            dec_fc_input = torch.cat([dec_fc_input, self.hidden[-1]], dim=1)

        dec_h = self.dec_action_fc(dec_fc_input)
        dec_mean = self.dec_action_mean(dec_h)

        if isinstance(self.dec_action_logvar, nn.Parameter):
            dec_logvar = self.dec_action_logvar
        else:
            dec_logvar = self.dec_action_logvar(dec_h)

        return Normal(dec_mean, dec_logvar)

    def reset_policy(self, labels=None, z=None, temperature=1.0, num_samples=0, device='cpu'):
        if self.requires_labels:
            assert labels is not None
            assert labels.size(-1) == self.config['label_dim']
    
            if z is not None:
                assert labels.size(0) == z.size(0)

            num_samples = labels.size(0)
            device = labels.device
            self.labels = labels

        if z is None:
            assert num_samples > 0
            assert device is not None
            z = torch.randn(num_samples, self.config['z_dim']).to(device)
        
        self.z = z
        self.temperature = temperature

        if self.is_recurrent:
            self.hidden = self.init_hidden_state(batch_size=z.size(0)).to(z.device)

    def prepare_stage(self, train_config):
        self.stage += 1
        self.init_optimizer(train_config['learning_rate']) 
        self.clip_grad = lambda : nn.utils.clip_grad_norm_(self.parameters(), train_config['clip'])

    def init_optimizer(self, lr, l2_penalty=0.0):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr, weight_decay=l2_penalty)

    def optimize(self, losses):
        assert isinstance(losses, dict)
        self.optimizer.zero_grad()
        total_loss = sum(losses.values())
        total_loss.backward()
        self.clip_grad()
        self.optimizer.step()

    def forward(self, states, actions, reconstruct=False, no_teacher_force=False):
        self.log.reset()

        assert actions.size(1)+1 == states.size(1) # final state has no corresponding action
        states = states.transpose(0,1)
        actions = actions.transpose(0,1)

        # Encoder outputs features that are passed through fully-connected layers which are 
        # then mapped to mean and logvar that define a normal distribution (the posterior) 
        posterior = self.encode(states[:-1], actions=actions)
        embedding = posterior.mean # we treat the mean of the resulting posterior as our embedding

        # Regularization loss is meant to force the posterior to be roughly near a unit
        # Gaussian distribution
        kld = Normal.kl_divergence(posterior, free_bits=0.0).detach()
        self.log.metrics['kl_div_true'] = torch.sum(kld)

        kld = Normal.kl_divergence(posterior, free_bits=1/self.config['z_dim'])
        self.log.losses['kl_div'] = torch.sum(kld)

		# Decode
        self.reset_policy(z=posterior.sample())

        # The reconstruction is created by rolling out new states
       	curr_state = states[0]
        if reconstruct:
            reconstruction = [curr_state]

        for t in range(actions.size(0)):
            # Create action distribution
            action_likelihood = self.decode_action(curr_state)
            
            # Reconstruction loss is the NLL of the true action under the 
            # predicted distribution of actions
            self.log.losses['nll'] -= action_likelihood.log_prob(actions[t])

            if self.is_recurrent:
                self.update_hidden(curr_state, actions[t])
            
            # If we aren't using teacher forcing, then we predict a distribution
            # of actions based on the previous state and the embedding. We only
            # provide the true state on the first time step. Each subsequent 
            # is the combination of an action sampled from the predicted 
            # distribution and the previous state.
            if no_teacher_force:
                curr_state = curr_state + action_likelihood.sample()
                if reconstruct:
                    reconstruction.append(curr_state)
            # If we are using teacher forcing, then the state used to predict
            # the action distribution at each time step is the true state, 
            # instead of a synthesized one. 
            else:
                if reconstruct:
                    reconstruction.append(curr_state + action_likelihood.sample())
                curr_state = states[t+1]

        if reconstruct:
            return self.log, torch.stack(reconstruction), posterior.mean

        return self.log
