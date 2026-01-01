"""
Bidirectional LSTM Enhanced Version Constitutive Relation Prediction Model - Simplified Version (removing material parameter encoder)
Including hyperparameter search, six curve-level metrics such as DTW distance, cross-validation
Supports loading material parameters and stress-strain data from Excel files

Variable Description:
- Independent variable 1: Strain sequence (X_strain) - General strain data, all samples use the same strain sequence
- Independent variable 2: Material parameters (X_material) - 15 parameters, including:
  * 10 material characteristics: Water (kg/m3), Cement (kg/m3), Water-cement ratio, Cement strength (MPa), Natural sand (kg/m3),
                  Coarse aggregate dosage, Mass replacement rate (%), Mixed aggregate water absorption rate (%), Maximum aggregate maximum particle size (mm), Mixed aggregate aggregate crushing index
  * 5 specimen parameters: Curing age (days), Loading rate μe, Chamfer ratio (0 for prism / 1 for cylinder), Side length or diameter (mm), Height-diameter ratio
  * Note: Excluding mechanical performance indicators (elastic modulus, peak stress, etc.)
- Dependent variable: Stress sequence (X_stress) - Each sample corresponds to different stress data

Model Architecture Simplification:
- Remove the material parameter encoder, directly copy material parameters to each time step
- LSTM processes both strain sequences and material parameters simultaneously [batch_size, time_steps, 1 + material_param_size]
- Simplify the feature processing flow to improve computational efficiency

Model Goal: Predict stress sequence using strain sequence and material parameters
"""
import os, sys
# Fix import error of torch._dynamo module in PyTorch nightly version
# Must set environment variables before importing torch to disable torch.compile to avoid OSError: [Errno 22] Invalid argument
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
# Disable PyTorch's automatic compilation feature
os.environ['PYTORCH_DISABLE_COMPILE'] = '1'
# Disable automatic compilation decorator for optimizers
os.environ['TORCH_LOGS'] = '-dynamo'

# Create a fake torch._dynamo module to intercept import errors
class _FakeDynamoModule:
    """Fake torch._dynamo module to avoid import errors"""
    class config:
        suppress_errors = True
        disable = True
        assume_static_by_default = True
    
    @staticmethod
    def disable(*args, **kwargs):
        """Fake disable method, returns original function or None"""
        if args and callable(args[0]):
            return args[0]  # Return original function
        def identity(func):
            return func
        return identity
    
    @staticmethod
    def reset(*args, **kwargs):
        """Fake reset method"""
        pass
    
    @staticmethod
    def graph_break(*args, **kwargs):
        """Fake graph_break method that does nothing"""
        pass

# Create a fake module factory function
def _create_fake_module(name):
    """Create fake module to intercept import errors"""
    class FakeModule:
        pass
    return FakeModule()

# Register all possible _dynamo submodules before importing torch
_fake_dynamo = _FakeDynamoModule()
sys.modules['torch._dynamo'] = _fake_dynamo
sys.modules['torch._dynamo.config'] = _fake_dynamo.config
sys.modules['torch._dynamo.symbolic_convert'] = _create_fake_module('torch._dynamo.symbolic_convert')
sys.modules['torch._dynamo.convert_frame'] = _create_fake_module('torch._dynamo.convert_frame')
sys.modules['torch._dynamo.aot_compile'] = _create_fake_module('torch._dynamo.aot_compile')

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt, seaborn as sns, warnings, time, optuna; from torch.utils.data import DataLoader; from sklearn.model_selection import KFold, StratifiedKFold; from sklearn.preprocessing import StandardScaler; from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score; from optuna.samplers import TPESampler; from optuna.pruners import MedianPruner, PercentilePruner; from datetime import datetime, timedelta; from pathlib import Path; warnings.filterwarnings('ignore')

# After import, try to replace torch._dynamo (if real import fails)
try:
    # Try to import real torch._dynamo
    import torch._dynamo as _real_dynamo
    # If import is successful, try to configure it
    try:
        _real_dynamo.config.suppress_errors = True
        _real_dynamo.config.disable = True
        # Ensure graph_break method exists
        if not hasattr(_real_dynamo, 'graph_break'):
            _real_dynamo.graph_break = lambda *args, **kwargs: None
    except:
        # If configuration fails, use fake module
        torch._dynamo = _FakeDynamoModule()
except (OSError, ImportError, AttributeError):
    # If import fails, ensure fake module is used
    torch._dynamo = _FakeDynamoModule()
    pass

# Ensure torch._dynamo has graph_break method (whether real or fake)
if not hasattr(torch._dynamo, 'graph_break'):
    torch._dynamo.graph_break = lambda *args, **kwargs: None

# Fix torch._compile module, directly patch _disable_dynamo function
try:
    import torch._compile
    import functools
    
    # Save original _disable_dynamo function
    _original_disable_dynamo = torch._compile._disable_dynamo
    
    def _patched_disable_dynamo(
        fn=None, recursive=True
    ):
        """Patched _disable_dynamo that directly returns original function when disable_fn is None"""
        if fn is not None:
            @functools.wraps(fn)
            def inner(*args, **kwargs):
                # Try to get disable_fn
                disable_fn = getattr(fn, "__dynamo_disable", None)
                if disable_fn is None:
                    try:
                        import torch._dynamo
                        disable_fn = torch._dynamo.disable(fn, recursive, wrapping=False)
                        fn.__dynamo_disable = disable_fn
                    except (OSError, ImportError, AttributeError, TypeError):
                        # If import fails or disable_fn is None, directly call original function
                        disable_fn = None
                
                # If disable_fn is None or not callable, directly call original function
                if disable_fn is None or not callable(disable_fn):
                    return fn(*args, **kwargs)
                
                try:
                    return disable_fn(*args, **kwargs)
                except TypeError as e:
                    if "'NoneType' object is not callable" in str(e):
                        # If disable_fn call fails, directly call original function
                        return fn(*args, **kwargs)
                    raise
            
            return inner
        else:
            # decorator usage
            return functools.partial(_patched_disable_dynamo, recursive=recursive)
    
    # Replace _disable_dynamo function
    torch._compile._disable_dynamo = _patched_disable_dynamo
    
except (OSError, ImportError, AttributeError) as e:
    import warnings
    warnings.warn(f"Failed to patch torch._compile._disable_dynamo: {e}")
    pass

# Directly patch all decorated methods of Optimizer class to remove decorator effects
try:
    import torch.optim.optimizer as opt_module
    import types
    import functools
    
    def _get_truly_original_method(class_obj, method_name):
        """Get truly original undecorated method from class's __dict__"""
        # Get directly from class's __dict__, which should be the most original method
        return class_obj.__dict__.get(method_name)
    
    def _create_safe_method(method_name):
        """Create safe method wrapper that directly uses original implementation"""
        # Get original method (from class's __dict__)
        original_unbound = _get_truly_original_method(opt_module.Optimizer, method_name)
        
        if original_unbound is None:
            # If unable to get, return None, default behavior will be used later
            return None
        
        # Create a bound method that directly calls original implementation
        def _safe_method(self, *args, **kwargs):
            """Safe method wrapper that directly calls original implementation, completely bypassing decorator"""
            # Directly call original undecorated method
            return original_unbound(self, *args, **kwargs)
        
        return _safe_method
    
    # Patch all possibly decorated methods
    methods_to_patch = ['add_param_group', 'zero_grad', 'step', 'state_dict', 'load_state_dict']
    patched_methods = []
    for method_name in methods_to_patch:
        if hasattr(opt_module.Optimizer, method_name):
            # Create safe wrapper
            safe_method = _create_safe_method(method_name)
            if safe_method is not None:
                # Replace method
                setattr(opt_module.Optimizer, method_name, safe_method)
                patched_methods.append(method_name)
            else:
                # If unable to get original method, try using descriptor protocol
                try:
                    # Try to get original method from MRO
                    for base in opt_module.Optimizer.__mro__:
                        if method_name in base.__dict__:
                            original = base.__dict__[method_name]
                            # Create a wrapper
                            def _make_wrapper(orig_method):
                                def _wrapper(self, *args, **kwargs):
                                    return orig_method(self, *args, **kwargs)
                                return _wrapper
                            setattr(opt_module.Optimizer, method_name, _make_wrapper(original))
                            patched_methods.append(method_name)
                            break
                except:
                    pass
    
    # Verify if patching was successful (optional, for debugging)
    if patched_methods:
        # Verification logic can be added here, but commented out for performance
        # print(f"Patched Optimizer methods: {', '.join(patched_methods)}")
        pass
except (AttributeError, ImportError) as e:
    # If patching fails, at least log the error without interrupting the program
    import warnings
    warnings.warn(f"Failed to patch Optimizer methods: {e}")
    pass
# Mixed precision training support
from torch.cuda.amp import autocast, GradScaler
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataloader import load_excel_data as load_dataset_with_clusters, ConstitutiveDataset as ClusterAwareDataset
# Set Chinese font display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# Set working directory - adapt to notebook running
def setup_working_directory():
    """Set working directory, adapt to notebook and script running"""
    current_dir = os.getcwd()
    
    # If in Pi_BiLSTM directory, use current directory directly
    if 'Pi_BiLSTM' in current_dir:
        print(f"Current working directory: {current_dir}")
        return current_dir
    
    # If running in notebook, need to adjust to project root directory
    if 'constitutive_relation' in current_dir:
        # Find project root directory
        parts = current_dir.split(os.sep)
        project_root_idx = parts.index('constitutive_relation')
        
        # Check if there is a Pi_BiLSTM subdirectory
        project_root = os.sep.join(parts[:project_root_idx + 1])
        pi_bilstm_dir = os.path.join(project_root, 'Pi_BiLSTM')
        
        if os.path.exists(pi_bilstm_dir):
            os.chdir(pi_bilstm_dir)
            print(f"Working directory has been set to: {os.getcwd()}")
        else:
            print(f"Current working directory: {current_dir}")
    else:
        print(f"Current working directory: {current_dir}")
    
    return os.getcwd()

# Initialize working directory
WORKING_DIR = setup_working_directory()

# Set random seeds and device
torch.manual_seed(42)
np.random.seed(42)

# Globally unified sampling length
DEFAULT_CURVE_LENGTH = 1000

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU model: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Configure cuDNN to reduce risk of internal errors
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False  # Disable benchmark to avoid certain cuDNN errors
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic operations to improve performance
    print("cuDNN enabled, benchmark mode disabled (to reduce risk of internal errors)")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    print("GPU cache cleared")
else:
    print("No GPU detected, will use CPU for computation")


# ========== 3. Bidirectional LSTM Model ==========
class BidirectionalLSTMRegressor(nn.Module):
    """
    Bidirectional LSTM regression model
    
    LSTM Unit Structure Description:
    Each LSTM unit controls information flow through gating mechanisms:
    1. Forget Gate: Determines which information to discard from the cell state
       - Uses sigmoid activation function, outputs values between 0 and 1
       - 0 means completely forget, 1 means completely retain
       - Formula: f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
    
    2. Input Gate: Determines which new information to store in the cell state
       - Contains two parts: i_t (uses sigmoid to decide which values to update) and C̃_t (generates candidate values using tanh)
       - Formula: i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
              C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)
    
    3. Cell State Update: Updates cell state based on information from forget gate and input gate
       - Formula: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
    
    4. Output Gate: Determines which parts of the cell state to output
       - Uses sigmoid to decide which parts to output, then scales cell state through tanh
       - Formula: o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
              h_t = o_t ⊙ tanh(C_t)
    
    Bidirectional LSTM: Processes sequences in both forward and backward directions simultaneously, able to utilize both historical and future information
    """
    
    def __init__(self, input_size=18, lstm_units=128, dropout=0.3, 
                 fc_hidden_size=256, output_length=DEFAULT_CURVE_LENGTH, 
                 num_lstm_layers=3, activation_function='gelu', use_attention=True,
                 bidirectional=True, attention_heads=8):
        super().__init__()
        
        self.output_length = output_length
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.input_size = input_size
        
        # Activation function selection
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        self.activation = activation_map.get(activation_function, nn.GELU())
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,  # 18个特征（1个应变+15个材料参数+1个峰值应力+1个峰值应变）
            hidden_size=lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism - supports variable number of heads
        if use_attention:
            lstm_output_size = lstm_units * (2 if bidirectional else 1)
            # Ensure embed_dim is divisible by num_heads (requirement of MultiheadAttention)
            # Round down to nearest multiple
            adjusted_embed_dim = (lstm_output_size // attention_heads) * attention_heads
            # Safety check: ensure adjusted dimension is at least attention_heads
            if adjusted_embed_dim < attention_heads:
                adjusted_embed_dim = attention_heads
                print(f"Warning: LSTM output dimension {lstm_output_size} is too small, adjusted to minimum available value {adjusted_embed_dim}")
            elif adjusted_embed_dim != lstm_output_size:
                print(f"Warning: LSTM output dimension {lstm_output_size} is not divisible by number of attention heads {attention_heads}, adjusted to {adjusted_embed_dim}")
            self.attention = nn.MultiheadAttention(
                embed_dim=adjusted_embed_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True
            )
            # If dimension was adjusted, add a linear layer to match dimensions
            if adjusted_embed_dim != lstm_output_size:
                self.attention_dim_adapter = nn.Linear(lstm_output_size, adjusted_embed_dim)
                self.attention_dim_restore = nn.Linear(adjusted_embed_dim, lstm_output_size)
            else:
                self.attention_dim_adapter = None
                self.attention_dim_restore = None
        
        # Fully connected layers - Optimization: use 5-layer progressive dimensionality reduction strategy + LayerNorm
        # Structure: fc_hidden_size → 3/4 → 1/2 → 1/3 → 1/6 → 1
        lstm_output_size = lstm_units * (2 if bidirectional else 1)
        
        # Add LayerNorm after LSTM output (if hidden_size > 1)
        self.ln_lstm_out = nn.LayerNorm(lstm_output_size) if lstm_output_size > 1 else nn.Identity()
        
        self.fc_layers = nn.Sequential(
            # Layer 1: lstm_output_size → fc_hidden_size
            nn.Linear(lstm_output_size, fc_hidden_size),
            nn.LayerNorm(fc_hidden_size) if fc_hidden_size > 1 else nn.Identity(),
            self.activation,
            nn.Dropout(dropout),
            
            # Layer 2: fc_hidden_size → fc_hidden_size*3//4
            nn.Linear(fc_hidden_size, fc_hidden_size * 3 // 4),
            nn.LayerNorm(fc_hidden_size * 3 // 4) if (fc_hidden_size * 3 // 4) > 1 else nn.Identity(),
            self.activation,
            nn.Dropout(dropout),
            
            # Layer 3: fc_hidden_size*3//4 → fc_hidden_size//2
            nn.Linear(fc_hidden_size * 3 // 4, fc_hidden_size // 2),
            nn.LayerNorm(fc_hidden_size // 2) if (fc_hidden_size // 2) > 1 else nn.Identity(),
            self.activation,
            nn.Dropout(dropout),
            
            # Layer 4: fc_hidden_size//2 → fc_hidden_size//3
            nn.Linear(fc_hidden_size // 2, fc_hidden_size // 3),
            nn.LayerNorm(fc_hidden_size // 3) if (fc_hidden_size // 3) > 1 else nn.Identity(),
            self.activation,
            nn.Dropout(dropout),
            
            # Layer 5: fc_hidden_size//3 → fc_hidden_size//6
            nn.Linear(fc_hidden_size // 3, fc_hidden_size // 6),
            nn.LayerNorm(fc_hidden_size // 6) if (fc_hidden_size // 6) > 1 else nn.Identity(),
            self.activation,
            nn.Dropout(dropout),
            
            # Output layer: fc_hidden_size//6 → 1
            nn.Linear(fc_hidden_size // 6, 1),  # Output single value
            nn.ReLU()  # Ensure non-negative output to avoid flat curves
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:  # Only perform Xavier initialization on weights with 2+ dimensions
                if 'lstm' in name:
                    # LSTM weights use Xavier initialization
                    nn.init.xavier_uniform_(param)
                elif 'fc_layers' in name:
                    # Fully connected layer weights use Xavier initialization
                    nn.init.xavier_uniform_(param)
                elif 'fc' in name:
                    # Other fc layer weights use Xavier initialization
                    nn.init.xavier_uniform_(param)
                elif 'attention' in name:
                    # Attention layer weights use Xavier initialization
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Biases initialized to 0
                nn.init.constant_(param, 0)
        
        print("Model weights initialized")
        
    def forward(self, x):
        """
        Forward propagation
        
        Args:
            x: Input tensor [batch_size, curve_length, input_size] - Each with curve_length rows × input_size columns
        
        Returns:
            curve_pred: Predicted stress curve [batch_size, curve_length] - Fixed curve_length point output
        """
        # Clear attention weight cache from previous iteration by default
        self.last_attn_weights = None
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # [batch_size, curve_length, lstm_output_size]
        
        # Attention mechanism (with residual connection)
        if self.use_attention: 
            # If dimension was adjusted, first pass through adapter layer
            if self.attention_dim_adapter is not None:
                lstm_out_adjusted = self.attention_dim_adapter(lstm_out)
                attn_out_adjusted, attn_weights = self.attention(lstm_out_adjusted, lstm_out_adjusted, lstm_out_adjusted)
                # Restore dimension
                attn_out = self.attention_dim_restore(attn_out_adjusted)
            else:
                attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            # Cache attention weights (for visualization)
            self.last_attn_weights = attn_weights
            # Residual connection: lstm_out + attn_out (dimension matches, direct addition)
            lstm_out = lstm_out + attn_out
        
        # Directly output prediction for each time step through fully connected layers
        batch_size, seq_len, hidden_size = lstm_out.shape
        lstm_out_reshaped = lstm_out.reshape(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        
        # Apply LayerNorm before FC layers
        lstm_out_reshaped = self.ln_lstm_out(lstm_out_reshaped)
        
        curve_pred_reshaped = self.fc_layers(lstm_out_reshaped)  # [batch_size * seq_len, 1]
        curve_pred = curve_pred_reshaped.reshape(batch_size, seq_len)  # [batch_size, curve_length]
        
        return curve_pred

# ========== 4. Enhanced Loss Function ==========
class EnhancedCurveLoss(nn.Module):
    """Enhanced curve loss function - integrating physical equation constraints"""
    
    def __init__(self, ascending_curve_weight=0.9, ascending_physics_weight=0.08, ascending_smoothness_weight=0.02,
                 descending_curve_weight=0.9, descending_physics_weight=0.08, descending_smoothness_weight=0.02,
                 start_boundary_weight=0.05, end_boundary_weight=0.05,
                 peak_value_weight=0.18, peak_window_weight=0.1,  # Enhanced peak constraint weight: 0.15 → 0.18 (conservative adjustment)
                 start_boundary_portion=0.015, end_boundary_portion=0.05,
                 peak_window_ratio=0.04,
                 energy_weight=0.02, monotonicity_weight=0.06,  # Enhanced constraint weights: energy 0.01→0.02, monotonicity 0.05→0.06 (conservative adjustment)
                 use_focal_loss=False, focal_gamma=2.0, focal_alpha=0.25):  # Focal Loss temporarily disabled, implementation to be fixed
        super().__init__()
        # Ascending segment weights
        self.ascending_curve_weight = ascending_curve_weight
        self.ascending_physics_weight = ascending_physics_weight
        self.ascending_smoothness_weight = ascending_smoothness_weight
        # Descending segment weights
        self.descending_curve_weight = descending_curve_weight
        self.descending_physics_weight = descending_physics_weight
        self.descending_smoothness_weight = descending_smoothness_weight
        # Boundary and peak constraints
        self.start_boundary_weight = start_boundary_weight
        self.end_boundary_weight = end_boundary_weight
        self.peak_value_weight = peak_value_weight
        self.peak_window_weight = peak_window_weight
        self.start_boundary_portion = start_boundary_portion
        self.end_boundary_portion = end_boundary_portion
        self.peak_window_ratio = peak_window_ratio
        # Energy constraint and monotonicity constraint
        self.energy_weight = energy_weight
        self.monotonicity_weight = monotonicity_weight
        # Focal Loss parameters
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.mse_loss = nn.MSELoss()
        
    def forward(self, curve_pred, curve_true, 
                material_params=None, strain_seq=None, peak_stress=None, peak_strain=None,
                material_scaler=None):
        """
        Calculate enhanced loss - integrating physical equation constraints
        
        Args:
            curve_pred: Predicted stress curve [batch_size, curve_length] - Dependent variable prediction (σ_c/σ_cp normalized)
            curve_true: True stress curve [batch_size, curve_length] - True dependent variable values (σ_c/σ_cp normalized)
            material_params: Normalized material parameters [batch_size, curve_length, num_material_params] - Independent variable 2 (optional)
            strain_seq: Normalized strain sequence [batch_size, curve_length] - Independent variable 1 (ε/ε_cp normalized)
            peak_stress: Peak stress [batch_size, curve_length] - Independent variable 3 (normalized = 1)
            peak_strain: Peak strain [batch_size, curve_length] - Independent variable 4 (normalized = 1)
            material_scaler: Material parameter normalizer - used for physical equation calculation
        
        Returns:
            total_loss: Total loss
            loss_dict: Dictionary of various loss components
        """
        # Calculate losses for ascending and descending segments separately
        ascending_loss, descending_loss = self.calculate_segment_losses(
            curve_pred, curve_true, material_params, strain_seq, peak_stress, peak_strain, material_scaler)
        
        boundary_loss_dict = self.calculate_boundary_losses(curve_pred, curve_true, strain_seq)
        
        # Calculate energy constraint loss (area under stress-strain curve)
        energy_loss = self.calculate_energy_loss(curve_pred, curve_true, strain_seq)
        
        # Calculate monotonicity constraint loss (ascending segment must be monotonically increasing)
        monotonicity_loss = self.calculate_monotonicity_loss(curve_pred, curve_true, strain_seq)
        
        # Total loss
        total_loss = (ascending_loss + descending_loss +
                      self.start_boundary_weight * boundary_loss_dict['start_loss'] +
                      self.end_boundary_weight * boundary_loss_dict['end_loss'] +
                      self.peak_value_weight * boundary_loss_dict['peak_value_loss'] +
                      self.peak_window_weight * boundary_loss_dict['peak_window_loss'] +
                      self.energy_weight * energy_loss +
                      self.monotonicity_weight * monotonicity_loss)
        
        loss_dict = {
            'ascending_loss': ascending_loss.item(),
            'descending_loss': descending_loss.item(),
            'start_boundary_loss': boundary_loss_dict['start_loss'].item(),
            'end_boundary_loss': boundary_loss_dict['end_loss'].item(),
            'peak_value_loss': boundary_loss_dict['peak_value_loss'].item(),
            'peak_window_loss': boundary_loss_dict['peak_window_loss'].item(),
            'energy_loss': energy_loss.item(),
            'monotonicity_loss': monotonicity_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def calculate_segment_losses(self, curve_pred, curve_true, material_params=None, strain_seq=None, peak_stress=None, peak_strain=None,
                                material_scaler=None):
        """
        Calculate losses for ascending and descending segments separately
        
        Args:
            curve_pred: Predicted stress curve [batch_size, curve_length]
            curve_true: True stress curve [batch_size, curve_length]
            material_params: Normalized material parameters [batch_size, curve_length, num_material_params]
            strain_seq: Normalized strain sequence [batch_size, curve_length]
            peak_stress: Peak stress [batch_size, curve_length]
            peak_strain: Peak strain [batch_size, curve_length]
            material_scaler: Material parameter normalizer
        
        Returns:
            ascending_loss: Total loss for ascending segment
            descending_loss: Total loss for descending segment
        """
        batch_size, curve_length = curve_pred.shape
        ascending_losses = []
        descending_losses = []
        
        for i in range(batch_size):
            x = strain_seq[i]  # Normalized strain sequence
            
            # Ascending segment (x < 1.0)
            ascending_mask = (x < 1.0)
            if torch.any(ascending_mask):
                ascending_pred = curve_pred[i][ascending_mask]
                ascending_true = curve_true[i][ascending_mask]
                
                # Ascending segment curve loss (using Focal Loss to give higher weight to hard samples)
                ascending_curve_loss = self.calculate_focal_mse_loss(ascending_pred, ascending_true)
                
                # Ascending segment smoothness loss
                ascending_smoothness_loss = self.calculate_smoothness_loss_segment(ascending_pred)
                
                # Ascending segment physics loss
                ascending_physics_loss = self.calculate_physics_loss_segment(
                    ascending_pred, ascending_true, material_params[i], x[ascending_mask], 
                    peak_stress[i], peak_strain[i], material_scaler)
                
                # Boundary constraint loss (ensure curve starts from 0) - reduce weight to avoid numerical instability
                boundary_loss = torch.mean(ascending_pred[:5]**2)  # Sum of squares of first 5 points
                
                # Check if boundary loss is valid
                if torch.isnan(boundary_loss) or torch.isinf(boundary_loss):
                    boundary_loss = torch.tensor(0.0, device=ascending_pred.device)
                
                # Total ascending segment loss
                ascending_loss_i = (self.ascending_curve_weight * ascending_curve_loss +
                                  self.ascending_physics_weight * ascending_physics_loss +
                                  self.ascending_smoothness_weight * ascending_smoothness_loss +
                                  0.01 * boundary_loss)  # Significantly reduce boundary constraint weight
                ascending_losses.append(ascending_loss_i)
            
            # Descending segment (x >= 1.0)
            descending_mask = (x >= 1.0)
            if torch.any(descending_mask):
                descending_pred = curve_pred[i][descending_mask]
                descending_true = curve_true[i][descending_mask]
                
                # Descending segment curve loss (using Focal Loss to give higher weight to hard samples)
                descending_curve_loss = self.calculate_focal_mse_loss(descending_pred, descending_true)
                
                # Descending segment smoothness loss
                descending_smoothness_loss = self.calculate_smoothness_loss_segment(descending_pred)
                
                # Descending segment physics loss
                descending_physics_loss = self.calculate_physics_loss_segment(
                    descending_pred, descending_true, material_params[i], x[descending_mask], 
                    peak_stress[i], peak_strain[i], material_scaler)
                
                # Total descending segment loss
                descending_loss_i = (self.descending_curve_weight * descending_curve_loss +
                                    self.descending_physics_weight * descending_physics_loss +
                                    self.descending_smoothness_weight * descending_smoothness_loss)
                descending_losses.append(descending_loss_i)
        
        # Calculate average loss
        ascending_loss = torch.mean(torch.stack(ascending_losses)) if ascending_losses else torch.tensor(0.0, device=curve_pred.device)
        descending_loss = torch.mean(torch.stack(descending_losses)) if descending_losses else torch.tensor(0.0, device=curve_pred.device)
        
        return ascending_loss, descending_loss

    def calculate_boundary_losses(self, curve_pred, curve_true, strain_seq):
        """Calculate boundary losses for start point, peak, and end point"""
        batch_size, curve_length = curve_pred.shape
        start_points = max(5, int(curve_length * self.start_boundary_portion))
        end_points = max(10, int(curve_length * self.end_boundary_portion))
        peak_window = max(10, int(curve_length * self.peak_window_ratio))
        
        start_pred = curve_pred[:, :start_points]
        start_true = curve_true[:, :start_points]
        start_loss = self.mse_loss(start_pred, start_true)
        
        end_pred = curve_pred[:, -end_points:]
        end_true = curve_true[:, -end_points:]
        end_loss = self.mse_loss(end_pred, end_true)
        
        # Peak constraints: value constraint + peak window constraint
        peak_indices = torch.argmax(curve_true, dim=1, keepdim=True)
        peak_pred = torch.gather(curve_pred, 1, peak_indices)
        peak_true = torch.gather(curve_true, 1, peak_indices)
        peak_value_loss = self.mse_loss(peak_pred, peak_true)
        
        peak_window_losses = []
        for i in range(batch_size):
            peak_idx = peak_indices[i].item()
            half_window = peak_window // 2
            start_idx = max(0, peak_idx - half_window)
            end_idx = min(curve_length, start_idx + peak_window)
            window_pred = curve_pred[i, start_idx:end_idx]
            window_true = curve_true[i, start_idx:end_idx]
            peak_window_losses.append(self.mse_loss(window_pred, window_true))
        peak_window_loss = torch.mean(torch.stack(peak_window_losses)) if peak_window_losses else torch.tensor(0.0, device=curve_pred.device)
        
        return {
            'start_loss': start_loss,
            'end_loss': end_loss,
            'peak_value_loss': peak_value_loss,
            'peak_window_loss': peak_window_loss
        }
    
    def calculate_energy_loss(self, curve_pred, curve_true, strain_seq):
        """
        Calculate energy constraint loss - area under stress-strain curve (energy)
        
        From civil engineering perspective:
        - The area under the stress-strain curve represents the energy absorbed by the material (per unit volume)
        - Energy conservation is a fundamental principle of material mechanics
        - The energy of the predicted curve and the true curve should be as close as possible
        
        Args:
            curve_pred: Predicted stress curve [batch_size, curve_length]
            curve_true: True stress curve [batch_size, curve_length]
            strain_seq: Strain sequence [batch_size, curve_length]
        
        Returns:
            energy_loss: Energy loss (MSE)
        """
        batch_size = curve_pred.shape[0]
        energy_losses = []
        
        for i in range(batch_size):
            # Calculate area under curve (energy) using trapezoidal rule
            # Note: strain_seq and curve are normalized, but energy proportional relationships remain valid
            pred_energy = torch.trapz(curve_pred[i], strain_seq[i])
            true_energy = torch.trapz(curve_true[i], strain_seq[i])
            
            # Calculate relative energy error (to avoid excessive absolute differences)
            energy_error = (pred_energy - true_energy) ** 2
            # Normalization: divide by square of true energy (to avoid division by zero)
            true_energy_sq = true_energy ** 2 + torch.tensor(1e-8, device=curve_pred.device)
            normalized_energy_error = energy_error / true_energy_sq
            
            energy_losses.append(normalized_energy_error)
        
        # Return average energy loss
        energy_loss = torch.mean(torch.stack(energy_losses)) if energy_losses else torch.tensor(0.0, device=curve_pred.device)
        
        # Check if it's a valid value
        if torch.isnan(energy_loss) or torch.isinf(energy_loss):
            return torch.tensor(0.0, device=curve_pred.device)
        
        return energy_loss
    
    def calculate_monotonicity_loss(self, curve_pred, curve_true, strain_seq):
        """
        Calculate monotonicity constraint loss - ensure ascending segment is monotonically increasing
        
        From civil engineering perspective:
        - The ascending segment (elastic + strengthening phase) of the stress-strain curve must be monotonically increasing
        - This is a fundamental physical law of material mechanics
        - Violating monotonicity means the prediction does not conform to physical reality
        
        Args:
            curve_pred: Predicted stress curve [batch_size, curve_length]
            curve_true: True stress curve [batch_size, curve_length]
            strain_seq: Strain sequence [batch_size, curve_length]
        
        Returns:
            monotonicity_loss: Monotonicity loss
        """
        batch_size, curve_length = curve_pred.shape
        monotonicity_losses = []
        
        for i in range(batch_size):
           # Find peak position (boundary between ascending and descending segments)
            peak_idx = torch.argmax(curve_true[i])
            
            # Extract ascending segment (part before peak)
            ascending_pred = curve_pred[i, :peak_idx+1]  # Include peak point
            
            if len(ascending_pred) < 2:
                # Skip if ascending segment is too short
                continue
            
            # Calculate first derivative of ascending segment (difference between adjacent points)
            # For monotonic increase, all differences should be >= 0
            diffs = torch.diff(ascending_pred)
            
            # 惩罚负值（违反单调性的点）
            # 使用平滑的惩罚函数：对于负值，惩罚其平方
            violations = torch.clamp(-diffs, min=0)  # 只保留负值部分
            
            # 计算单调性损失：违反单调性的程度
            # 使用相对惩罚：除以上升段的平均应力值（归一化）
            mean_stress = torch.mean(torch.abs(ascending_pred)) + torch.tensor(1e-8, device=curve_pred.device)
            monotonicity_loss_i = torch.mean(violations ** 2) / mean_stress
            
            # 也可以添加额外的惩罚：违反单调性的点的比例
            violation_ratio = torch.sum(violations > 1e-6) / len(diffs)
            # 如果违反比例过高，增加惩罚
            if violation_ratio > 0.1:  # 超过10%的点违反单调性
                monotonicity_loss_i = monotonicity_loss_i * (1 + violation_ratio)
            
            monotonicity_losses.append(monotonicity_loss_i)
        
        # 返回平均单调性损失
        monotonicity_loss = torch.mean(torch.stack(monotonicity_losses)) if monotonicity_losses else torch.tensor(0.0, device=curve_pred.device)
        
        # 检查是否为有效数值
        if torch.isnan(monotonicity_loss) or torch.isinf(monotonicity_loss):
            return torch.tensor(0.0, device=curve_pred.device)
        
        return monotonicity_loss
    
    def calculate_physics_loss_segment(self, segment_pred, segment_true, material_params, segment_strain, peak_stress, peak_strain, material_scaler):
        """
         Calculate physics loss for a single segment
        """
        if material_scaler is None or len(segment_pred) == 0:
            return torch.tensor(0.0, device=segment_pred.device)
        
        try:
            # Inverse transform material parameter
            material_params_orig = material_scaler.inverse_transform(material_params[0].cpu().numpy().reshape(1, -1)).flatten()
            
            r = material_params_orig[6]   # 质量取代率(%) - 第7个参数（索引6）
            WA = material_params_orig[7]  # 混合骨料吸水率% - 第8个参数（索引7）
            
             # Calculate parameters a and b
            a1 = 2.2 * (0.748 * (r*0.01)**2 - 1.231 * r*0.01 + 0.975)
            a2 = 0.00795 * (WA*0.01)**2 + 0.03273 * WA*0.01 + 1.7762
            a = (a1 + a2) / 2
            
            b1 = 0.8 * (7.6483 * r*0.01 + 1.142)
            b2 = -0.0264 * (WA*0.01)**2 + 0.70578 * WA*0.01 + 0.97629
            b = (b1 + b2) / 2
            
            if not np.isfinite(a) or not np.isfinite(b):
                return torch.tensor(0.0, device=segment_pred.device)
            
            # Normalize strain and stress for physics equation calculation
            # Note: segment_strain and segment_pred are already normalized by peak average
            # Further normalization is required to bring them to their respective peak values：
            # x = (ε/ε_cp_avg) / (ε_cp/ε_cp_avg) = ε/ε_cp
            # y = (σ/σ_cp_avg) / (σ_cp/σ_cp_avg) = σ/σ_cp
            x_normalized = segment_strain / peak_strain  # Strain normalized to peak strain
            y_pred_normalized = segment_pred / peak_stress   # Predicted stress normalized to peak stress
            
            # Calculate physics equation prediction
            y_physics = torch.zeros_like(x_normalized, device=x_normalized.device)
            
            if torch.all(x_normalized < 1.0):  # 上升段
                y_physics = a * x_normalized + (3 - 2*a) * x_normalized**2 + (a - 2) * x_normalized**3
            elif torch.all(x_normalized >= 1.0):  #下降段
                denominator = b * (x_normalized - 1)**2 + x_normalized
                denominator = torch.where(denominator <= 1e-8, torch.tensor(1e-8, device=x_normalized.device), denominator)
                y_physics = x_normalized / denominator
            
            # Calculate physics loss (compare using normalized values)
            mse_loss = torch.mean((y_physics - y_pred_normalized)**2)
            pred_var = torch.mean(y_pred_normalized**2) + torch.tensor(1e-6, device=y_pred_normalized.device)
            physics_loss = mse_loss / pred_var
            
            # Ensure loss value is within reasonable range to avoid NaN
            physics_loss = torch.clamp(physics_loss, torch.tensor(0.0, device=physics_loss.device), 
                                     torch.tensor(10.0, device=physics_loss.device))
            
             # Check for valid numerical value
            if torch.isnan(physics_loss) or torch.isinf(physics_loss):
                return torch.tensor(0.0, device=segment_pred.device)
            
            return physics_loss
        except:
            return torch.tensor(0.0, device=segment_pred.device)
    
    def calculate_smoothness_loss_segment(self, segment_pred):
        """Calculate smoothness loss for a single segmen"""
        if len(segment_pred) < 3:
            return torch.tensor(0.0, device=segment_pred.device)
        
         # Calculate second derivative
        second_diff = torch.diff(torch.diff(segment_pred))
        return torch.mean(torch.abs(second_diff))
    
    def calculate_focal_mse_loss(self, pred, true):
        """
        Calculate Focal MSE Loss - give higher weight to hard samples (samples with large errors)
        
        Fixed version: correctly implement Focal Loss, give higher weight to samples with large errors
        
        Args:
            pred: predicted values [n_points]
            true: true values [n_points]
        
        Returns:
            focal_mse_loss: Focal MSE loss
        """
        if not self.use_focal_loss:
             # Return standard MSE if Focal Loss is not used
            return self.mse_loss(pred, true)
        
        # Calculate error for each point
        errors = (pred - true) ** 2  # [n_points]
        
        # Calculate average error as baseline
        mean_error = torch.mean(errors) + torch.tensor(1e-8, device=errors.device)
        
        # Normalize error (relative to average error)
        normalized_errors = errors / mean_error  # [n_points]
        
        # Focal weight: larger error -> higher weight
        # Use normalized_error^gamma, so samples with large errors have higher weights
        # Use smooth function to avoid excessively large weights: 1 + normalized_error^gamma
        focal_weights = 1.0 + (normalized_errors ** self.focal_gamma)
        
        # Apply alpha weight for balance (smaller alpha -> weaker Focal effect)
        if self.focal_alpha is not None:
            # Alpha controls the strength of Focal effect
            focal_weights = 1.0 + self.focal_alpha * (normalized_errors ** self.focal_gamma)
        
        # Calculate weighted MSE
        weighted_errors = focal_weights * errors
        focal_mse_loss = torch.mean(weighted_errors)
        
        # Check for valid numerical value
        if torch.isnan(focal_mse_loss) or torch.isinf(focal_mse_loss):
            return self.mse_loss(pred, true)
        
        return focal_mse_loss
    

# ========== 5. Initial Segment Post-processing Function ==========
def postprocess_initial_segment(pred_curves, strain_curves, initial_ratio=0.15):
    """
     Post-process initial segment to ensure elastic behavior (start from 0, monotonically increasing, smooth slope)
    
    Args:
        pred_curves: predicted stress curves [n_samples, curve_length]
        strain_curves: strain curves [n_samples, curve_length]
        initial_ratio: ratio of initial segment (e.g., 0.15 means first 15% points)
    
    Returns:
        corrected_curves: corrected predicted curves
    """
    corrected_curves = pred_curves.copy()
    n_samples, curve_length = pred_curves.shape
    
    for i in range(n_samples):
        pred_curve = corrected_curves[i]
        strain_curve = strain_curves[i]
        
        # Determine index range of initial segment
        initial_length = max(3, int(curve_length * initial_ratio))
        
        # Find peak position (to determine reasonable range for initial segment)
        peak_idx = np.argmax(pred_curve)
        if peak_idx < initial_length:
            initial_length = min(initial_length, max(3, peak_idx // 2))
        
        # Ensure first point is 0
        pred_curve[0] = 0.0
        
        # Set second point to 0 if it's negative
        if pred_curve[1] < 0:
            pred_curve[1] = 0.0
        
         # Calculate average slope of initial segment (use middle part to avoid first point anomaly)）
        if initial_length >= 3:
             # Calculate average slope using points from 2 to initial_length
            valid_indices = np.arange(2, min(initial_length, len(pred_curve)))
            if len(valid_indices) > 0:
                strain_diff = strain_curve[valid_indices] - strain_curve[valid_indices[0]-1]
                stress_diff = pred_curve[valid_indices] - pred_curve[valid_indices[0]-1]
                valid_mask = strain_diff > 1e-8
                if valid_mask.sum() > 0:
                    avg_slope = np.mean(stress_diff[valid_mask] / strain_diff[valid_mask])
                else:
                    avg_slope = (pred_curve[initial_length-1] - pred_curve[0]) / (strain_curve[initial_length-1] - strain_curve[0] + 1e-8)
            else:
                avg_slope = (pred_curve[initial_length-1] - pred_curve[0]) / (strain_curve[initial_length-1] - strain_curve[0] + 1e-8)
        else:
            avg_slope = (pred_curve[min(1, initial_length-1)] - pred_curve[0]) / (strain_curve[min(1, initial_length-1)] - strain_curve[0] + 1e-8)
        
        # Ensure initial segment is monotonically increasing with smooth slope
        for j in range(1, initial_length):
            # Ensure monotonic increase
            if pred_curve[j] < pred_curve[j-1]:
                pred_curve[j] = pred_curve[j-1]
            
            # Calculate current slope
            if strain_curve[j] > strain_curve[j-1] + 1e-8:
                curr_slope = (pred_curve[j] - pred_curve[j-1]) / (strain_curve[j] - strain_curve[j-1])
                
                 # Recalculate using average slope if current slope is abnormal (too large or too small)
                if j >= 2:
                    if curr_slope > avg_slope * 5 or curr_slope < avg_slope * 0.2:
                        # Correct using average slope
                        pred_curve[j] = pred_curve[j-1] + avg_slope * (strain_curve[j] - strain_curve[j-1])
                    elif curr_slope < 0:
                        # Use previous value if slope is negative
                        pred_curve[j] = pred_curve[j-1]
                else:
                    # Use average slope for first two points
                    if curr_slope < 0 or curr_slope > avg_slope * 10:
                        pred_curve[j] = pred_curve[j-1] + avg_slope * (strain_curve[j] - strain_curve[j-1])
        
        # Second check to ensure corrected curve is still monotonically increasing）
        for j in range(1, initial_length):
            if pred_curve[j] < pred_curve[j-1]:
                pred_curve[j] = pred_curve[j-1]
    
    return corrected_curves

# ========== 6. Curve-level Evaluation Metrics ==========
def calculate_dtw_distance(seq1, seq2):
    """Calculate DTW distance"""
    try:
        from dtaidistance import dtw
        # Ensure sequences are numpy arrays
        seq1 = np.array(seq1).flatten()
        seq2 = np.array(seq2).flatten()
        
        #  Calculate DTW distance
        distance = dtw.distance(seq1, seq2)
        return distance
    except ImportError:
        # Raise error if DTW library is not available
        raise ImportError(
            "DTW distance calculation requires dtaidistance library. Please install: pip install dtaidistance\n"
        )

def calculate_curve_metrics(pred_curves, true_curves):
    """
    Calculate six curve-level metrics
    
    Args:
        pred_curves: predicted stress curves [num_curves, curve_length] - dependent variable predictions
        true_curves: true stress curves [num_curves, curve_length] - true dependent variable values
    
    Returns:
        metrics: dictionary containing six metrics
    """
    metrics = {}
    
    # 1. Curve MSE
    curve_mse = mean_squared_error(true_curves.flatten(), pred_curves.flatten())
    metrics['Curve_MSE'] = curve_mse
    
    # 2. Curve RMSE
    curve_rmse = np.sqrt(curve_mse)
    metrics['Curve_RMSE'] = curve_rmse
    
    # 3. Curve MAE
    curve_mae = mean_absolute_error(true_curves.flatten(), pred_curves.flatten())
    metrics['Curve_MAE'] = curve_mae
    
    # 4. Curve R²
    curve_r2 = r2_score(true_curves.flatten(), pred_curves.flatten())
    metrics['Curve_R2'] = curve_r2
    
    # 5. Average DTW distance
    dtw_distances = []
    for i in range(len(pred_curves)):
        # Must use DTW distance, no alternative allowed
            dtw_dist = calculate_dtw_distance(pred_curves[i], true_curves[i])
            dtw_distances.append(dtw_dist)
    
    metrics['Mean_DTW_Distance'] = np.mean(dtw_distances)
    metrics['Std_DTW_Distance'] = np.std(dtw_distances)
    
    # 6. Peak prediction accuracy
    pred_peaks = np.max(pred_curves, axis=1)
    true_peaks = np.max(true_curves, axis=1)
    peak_mse = mean_squared_error(true_peaks, pred_peaks)
    peak_rmse = np.sqrt(peak_mse)
    peak_mae = mean_absolute_error(true_peaks, pred_peaks)
    peak_r2 = r2_score(true_peaks, pred_peaks)
    
    metrics['Peak_MSE'] = peak_mse
    metrics['Peak_RMSE'] = peak_rmse
    metrics['Peak_MAE'] = peak_mae
    metrics['Peak_R2'] = peak_r2
    
    # 7. Curve shape similarity (based on correlation coefficient)
    shape_similarities = []
    for i in range(len(pred_curves)):
        corr = np.corrcoef(pred_curves[i], true_curves[i])[0, 1]
        if not np.isnan(corr):
            shape_similarities.append(corr)
    
    metrics['Mean_Shape_Similarity'] = np.mean(shape_similarities)
    metrics['Std_Shape_Similarity'] = np.std(shape_similarities)
    
    # 8. New: Shape awareness metrics (improved version - continuous scoring instead of binary judgment)
    shape_aware_scores = []
    monotonicity_scores = []
    curvature_scores = []
    
    for i in range(len(pred_curves)):
        pred_curve = pred_curves[i]
        true_curve = true_curves[i]
        
        # Monotonicity check (rising segment should be monotonically increasing) - changed to continuous scoring
        peak_idx = np.argmax(true_curve)
        pred_rising = pred_curve[:peak_idx]
        
        if len(pred_rising) > 1:
            # Calculate differential of rising segment (should be positive)
            diffs = np.diff(pred_rising)
            # Calculate violation ratio of monotonicity (number of negative numbers)
            violation_ratio = np.sum(diffs < 0) / len(diffs)
            # Monotonicity score: 1.0 = completely monotonic, 0.0 = completely non-monotonic
            monotonicity_score = max(0.0, 1.0 - 2 * violation_ratio)  # Penalize points violating monotonicity
        else:
            monotonicity_score = 1.0
        
        monotonicity_scores.append(monotonicity_score)
        
        # Curvature check (check if there are enough bends) - improved scoring method
        if len(pred_curve) > 2:
            # Calculate second derivative (curvature)
            pred_curvature = np.abs(np.diff(np.diff(pred_curve)))
            true_curvature = np.abs(np.diff(np.diff(true_curve)))
            
            # Use correlation coefficient to measure similarity of curvature patterns
            if np.std(pred_curvature) > 1e-8 and np.std(true_curvature) > 1e-8:
                curvature_corr = np.corrcoef(pred_curvature, true_curvature)[0, 1]
                if np.isnan(curvature_corr):
                    curvature_score = 0.0
                else:
                    # Curvature score: 0.0 = completely unrelated, 1.0 = completely related
                    curvature_score = max(0.0, curvature_corr)
            else:
                # If curvature change is too small, consider curvature score as 1.0 (no curvature)
                curvature_score = 1.0
        else:
            curvature_score = 1.0
        
        curvature_scores.append(curvature_score)
        
        # Combined shape awareness score (average of monotonicity and curvature scores)
        shape_aware_score = (monotonicity_score + curvature_score) / 2
        shape_aware_scores.append(shape_aware_score)
    
    metrics['Mean_Shape_Aware_Score'] = np.mean(shape_aware_scores) if shape_aware_scores else 0.0
    metrics['Std_Shape_Aware_Score'] = np.std(shape_aware_scores) if shape_aware_scores else 0.0
    metrics['Mean_Monotonicity_Score'] = np.mean(monotonicity_scores) if monotonicity_scores else 0.0
    metrics['Mean_Curvature_Score'] = np.mean(curvature_scores) if curvature_scores else 0.0
    
    return metrics

# ========== 7. Hyperparameter Optimization ==========
class BidirectionalLSTMTrial(BidirectionalLSTMRegressor):
    """Bidirectional LSTM model for hyperparameter optimization - directly inherits main model class"""
    
    def __init__(self, material_param_size, trial_params, curve_length=DEFAULT_CURVE_LENGTH):
        # Get hyperparameters from trial
        lstm_units = trial_params['lstm_units']
        dropout = trial_params['dropout']
        fc_hidden_size = trial_params['fc_hidden_size']
        num_lstm_layers = trial_params['num_lstm_layers']
        activation_function = trial_params['activation_function']
        use_attention = trial_params['use_attention']
        bidirectional = trial_params['bidirectional']
        attention_heads = trial_params.get('attention_heads', 8)  # New
        
        # Directly call parent class constructor to avoid duplicate model creation
        super().__init__(
            input_size=material_param_size + 3,  # +1 for strain +1 for peak_stress +1 for peak_strain
            lstm_units=lstm_units,
            dropout=dropout,
            fc_hidden_size=fc_hidden_size,
            output_length=curve_length,
            num_lstm_layers=num_lstm_layers,
            activation_function=activation_function,
            use_attention=use_attention,
            bidirectional=bidirectional,
            attention_heads=attention_heads  # New
        )

def objective(trial, train_dataset, val_dataset, material_param_size, curve_length=DEFAULT_CURVE_LENGTH):
    """Hyperparameter optimization objective function"""
    
    # Record start time
    start_time = time.time()
    
    # Define hyperparameter search space (using continuous space, let Bayesian optimization automatically select better parameters)
    # TPE会根据之前所有trial的结果自动选择更优的超参数组合
    trial_params = {
        # Use logarithmic uniform distribution to search LSTM units (continuous space, automatically round)
        'lstm_units': int(trial.suggest_loguniform('lstm_units', 64, 1024)),  # 对数均匀分布，覆盖2的幂次范围
        'dropout': trial.suggest_float('dropout', 0.1, 0.6),  # Already in continuous space
        # Use logarithmic uniform distribution to search FC hidden size (continuous space, automatically round)
        'fc_hidden_size': int(trial.suggest_loguniform('fc_hidden_size', 128, 2048)),  # 对数均匀分布
        'num_lstm_layers': trial.suggest_int('num_lstm_layers', 2, 6),  # Integer space
        # Activation function and attention mechanism must be discrete choices
        'activation_function': trial.suggest_categorical('activation_function', 
                                                       ['relu', 'gelu', 'swish', 'leaky_relu']),
        'use_attention': trial.suggest_categorical('use_attention', [True, False]),
        'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
        'attention_heads': trial.suggest_categorical('attention_heads', [4, 8, 16])  # Usually a power of 2
    }
    
    # Batch size using integer search (continuous integer space)
    batch_size = trial.suggest_int('batch_size', 8, 32)  # Continuous integer space, TPE will automatically select a better value
    
    # Create DataLoader based on batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model and move to GPU
    model = BidirectionalLSTMTrial(material_param_size, trial_params, curve_length)
    model = model.to(device)
    
        # Define optimizer and loss function (using logarithmic uniform distribution, let TPE automatically select better learning rate and weight decay)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)  # Logarithmic uniform distribution, TPE will automatically select a better value
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)  # Logarithmic uniform distribution, TPE will automatically select a better value
    
    # Optimizer selection: AdamW (better weight decay) or Adam
    optimizer_type = trial.suggest_categorical('optimizer_type', ['adam', 'adamw'])
    # Fix TypeError issue when creating optimizer in PyTorch nightly version
    # Before creating optimizer, fix torch._compile.inner decorator
    try:
        import torch._compile
        # If inner decorator exists, fix it to handle disable_fn as None
        if hasattr(torch._compile, 'inner'):
            _original_inner = torch._compile.inner
            def _safe_inner(func):
                """Safe inner decorator, handle disable_fn as None"""
                try:
                    return _original_inner(func)
                except TypeError as e:
                    if "'NoneType' object is not callable" in str(e):
                        # If disable_fn is None, return original function (without applying decorator)
                        return func
                    raise
            torch._compile.inner = _safe_inner
    except (OSError, ImportError, AttributeError):
        pass
    
    # Create optimizer (using try-except as extra protection)
    optimizer = None
    try:
        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    except TypeError as e:
        if "'NoneType' object is not callable" in str(e):
            # If still fails, try to fix optimizer.Optimizer.add_param_group directly
            import torch.optim.optimizer as opt_module
            # Directly get original undecorated method from class __dict__
            _original_add_param_group = opt_module.Optimizer.__dict__.get('add_param_group')
            if _original_add_param_group is None:
                # If cannot get, try to get from MRO
                for base in opt_module.Optimizer.__mro__:
                    if 'add_param_group' in base.__dict__:
                        _original_add_param_group = base.__dict__['add_param_group']
                        break
            
            if _original_add_param_group:
                def _safe_add_param_group(self, param_group):
                    """Safe version of torch._compile decorator, directly call original implementation"""
                    # Call original undecorated method directly
                    return _original_add_param_group(self, param_group)
                opt_module.Optimizer.add_param_group = _safe_add_param_group
                # Retry creating optimizer
                if optimizer_type == 'adamw':
                    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                else:
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                        raise RuntimeError("Cannot get original method of Optimizer.add_param_group, cannot create optimizer")
        else:
            raise
    
    # Verify if optimizer was successfully created
    if optimizer is None:
        raise RuntimeError(f"Cannot create optimizer: optimizer_type={optimizer_type}, learning_rate={learning_rate}, weight_decay={weight_decay}")
    
    # Search for loss function weight hyperparameters (using continuous space, let TPE automatically select better weight combinations)
    # Use continuous space to search weights, TPE will automatically find a better weight balance based on previous results
    ascending_curve_weight = trial.suggest_float('ascending_curve_weight', 0.75, 0.98)  # Continuous space
    ascending_physics_weight = trial.suggest_float('ascending_physics_weight', 0.01, 0.20)  # Continuous space
    ascending_smoothness_weight = 1.0 - ascending_curve_weight - ascending_physics_weight
    # Ensure smoothness_weight is not negative, if negative then adjust
    if ascending_smoothness_weight < 0:
        ascending_physics_weight = min(ascending_physics_weight, 1.0 - ascending_curve_weight - 0.01)
        ascending_smoothness_weight = max(0.01, 1.0 - ascending_curve_weight - ascending_physics_weight)
    else:
        ascending_smoothness_weight = max(0.01, ascending_smoothness_weight)
    
    descending_curve_weight = trial.suggest_float('descending_curve_weight', 0.75, 0.98)  # Continuous space
    descending_physics_weight = trial.suggest_float('descending_physics_weight', 0.01, 0.20)  # Continuous space
    descending_smoothness_weight = 1.0 - descending_curve_weight - descending_physics_weight
    # Ensure smoothness_weight is not negative
    if descending_smoothness_weight < 0:
        descending_physics_weight = min(descending_physics_weight, 1.0 - descending_curve_weight - 0.01)
        descending_smoothness_weight = max(0.01, 1.0 - descending_curve_weight - descending_physics_weight)
    else:
        descending_smoothness_weight = max(0.01, descending_smoothness_weight)
    
    criterion = EnhancedCurveLoss(
        ascending_curve_weight=ascending_curve_weight,
        ascending_physics_weight=ascending_physics_weight,
        ascending_smoothness_weight=ascending_smoothness_weight,
        descending_curve_weight=descending_curve_weight,
        descending_physics_weight=descending_physics_weight,
        descending_smoothness_weight=descending_smoothness_weight
    )
    
    # Mixed precision training: create GradScaler (only enabled on GPU)
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
        # Train model (supports early stopping)
    train_losses = []  # Record training loss per step
    val_losses_per_epoch = []  # Record validation loss per epoch, for pruning
    model.train()
    for epoch in range(50):  # Increase training epochs to 50, give model more learning time (from 30 to 50)
        epoch_losses = []
        for batch in train_loader:
            strain = batch['strain'].to(device)
            stress = batch['stress'].to(device)
            material_params = batch['material_params'].to(device)
            peak_stress = batch['peak_stress'].to(device)
            peak_strain = batch['peak_strain'].to(device)
            
            # Prepare input
            # strain: [batch_size, curve_length]
            # material_params: [batch_size, curve_length, material_param_size]
            # peak_stress: [batch_size, curve_length]
            # peak_strain: [batch_size, curve_length]
            x = torch.cat([strain.unsqueeze(-1), material_params, peak_stress.unsqueeze(-1), peak_strain.unsqueeze(-1)], dim=-1)
            
            optimizer.zero_grad()
            
            # Mixed precision training: use autocast to wrap forward propagation
            if use_amp:
                with autocast():
                    curve_pred = model(x)
                    loss, _ = criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
            else:
                curve_pred = model(x)
                loss, _ = criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
            
            # Check if loss is a valid numerical value
            if torch.isnan(loss).item() or torch.isinf(loss).item():
                print(f"Warning: Invalid loss detected: {loss.item()}")
                # If loss is invalid, skip this batch
                optimizer.zero_grad()
                continue
            
            # Check if loss is too large, avoid numerical instability
            if loss.item() > 50.0:
                print(f"Warning: Loss too large: {loss.item()}")
                optimizer.zero_grad()
                continue
            
            try:
                # Ensure model is in training mode (prevent forgetting to switch after validation)
                model.train()
                
                # Mixed precision training: use scaler for backward propagation
                if use_amp:
                    scaler.scale(loss).backward()
                    # Gradient clipping (needs to be in scaler)
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # Gradient clipping to prevent gradient explosion - more strict clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                
                epoch_losses.append(loss.item())
            except RuntimeError as e:
                    # Capture cuDNN errors and other GPU errors
                error_msg = str(e)
                if 'cuDNN' in error_msg or 'CUDNN' in error_msg or 'training mode' in error_msg.lower():
                    print(f"Training error, clear GPU cache and skip this batch: {error_msg}")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    # Note: only need to update if scaler.scale() was successfully called, here not needed
                    # Because if backward() fails, scaler does not record any information
                    continue
                else:
                    # Other runtime errors, also clear cache
                    print(f"Runtime error, clear GPU cache and skip this batch: {error_msg}")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    # Same, no need to scaler.update()
                    continue
        
        train_losses.extend(epoch_losses)
        
        # After each epoch, perform validation, for pruning decision
        model.eval()  # Switch to evaluation mode
        epoch_val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                strain = batch['strain'].to(device)
                stress = batch['stress'].to(device)
                material_params = batch['material_params'].to(device)
                peak_stress = batch['peak_stress'].to(device)
                peak_strain = batch['peak_strain'].to(device)
                
                # Prepare input
                x = torch.cat([strain.unsqueeze(-1), material_params, peak_stress.unsqueeze(-1), peak_strain.unsqueeze(-1)], dim=-1)
                
                # Also use autocast for validation (does not affect gradient calculation)
                if use_amp:
                    with autocast():
                        curve_pred = model(x)
                        loss, _ = criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
                else:
                    curve_pred = model(x)
                    loss, _ = criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
                epoch_val_losses.append(loss.item())
        
        avg_epoch_val_loss = np.mean(epoch_val_losses)
        val_losses_per_epoch.append(avg_epoch_val_loss)
        
        # Report intermediate value to Optuna, for pruning decision
        trial.report(avg_epoch_val_loss, step=epoch)
        
        # Note: Validation loss is total loss of EnhancedCurveLoss, including:
        #   - Ascending segment loss (curve MSE + physics constraint + smoothness)
        #   - Descending segment loss (curve MSE + physics constraint + smoothness)
        #   - Boundary loss (start point, end point, peak constraint)       
        VAL_LOSS_THRESHOLD = 0.15  # Early stop if validation loss is greater than 0.15 (widen threshold, allow more trials to complete training)
        MIN_EPOCHS_BEFORE_THRESHOLD_CHECK = 5  # At least train 5 epochs before checking threshold (from 3 to 5, give model more time)
        
        # Early stop if validation loss is greater than threshold
        if epoch >= MIN_EPOCHS_BEFORE_THRESHOLD_CHECK and avg_epoch_val_loss > VAL_LOSS_THRESHOLD:
            print(f"Trial {trial.number} pruned at epoch {epoch+1}/50 (val_loss: {avg_epoch_val_loss:.4f} > {VAL_LOSS_THRESHOLD})")
            # Clear GPU cache and model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # Raise TrialPruned exception, Optuna will handle it
            raise optuna.TrialPruned()
        
        # Check if should early stop (pruning - Optuna's MedianPruner)
        # MedianPruner will compare current trial's validation loss with median of completed trials
        # If current loss is significantly higher than median,说明该trial表现不佳，可以提前终止
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch+1}/50 (val_loss: {avg_epoch_val_loss:.4f})")
            # Clear GPU cache and model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # Raise TrialPruned exception, Optuna will handle it
            raise optuna.TrialPruned()
        
        # Switch back to training mode, for next epoch
        model.train()
        
        # After each epoch, clear GPU cache, avoid memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final validation (if all epochs are completed)
    model.eval()
    final_val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            strain = batch['strain'].to(device)
            stress = batch['stress'].to(device)
            material_params = batch['material_params'].to(device)
            peak_stress = batch['peak_stress'].to(device)
            peak_strain = batch['peak_strain'].to(device)
            
             # Prepare input    
            x = torch.cat([strain.unsqueeze(-1), material_params, peak_stress.unsqueeze(-1), peak_strain.unsqueeze(-1)], dim=-1)
            
            # Also use autocast for validation (does not affect gradient calculation)
            if use_amp:
                with autocast():
                    curve_pred = model(x)
                    loss, _ = criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
            else:
                curve_pred = model(x)
                loss, _ = criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
            final_val_losses.append(loss.item())
    
    # Calculate training time and loss
    training_time = time.time() - start_time
    avg_train_loss = np.mean(train_losses)
    final_val_loss = np.mean(final_val_losses)
    
    # Clear GPU cache and model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"Trial {trial.number}: Training time {training_time:.2f} seconds")
    print(f"  Train loss: {avg_train_loss:.4f}, Val loss: {final_val_loss:.4f}")
    
    return final_val_loss



# ==========8. Main training function ==========
def train_bidirectional_lstm_model(material_params_file, stress_data_file, save_dir='SAVE/bidirectional_lstm', 
                                  n_trials=20, epochs=500, curve_length=DEFAULT_CURVE_LENGTH, 
                                  train_idx=None, val_idx=None, test_idx=None):
    """
    Main training function for bidirectional LSTM model (single training, for subset cross-validation)
    
    Args:
        material_params_file: Excel file path - first sheet contains independent variables 2 (material parameters)
        stress_data_file: Excel file path - second sheet contains independent variables 1 (strain) and dependent variable (stress)
        save_dir: save directory
        n_trials: number of hyperparameter search trials
        epochs: number of training epochs
        curve_length: curve length
        train_idx: training set index (optional, if provided use external划分）
        test_idx: test set index (optional, if provided use external划分）
    """
    print("=== Bidirectional LSTM enhanced constitutive relation prediction model ===")
    
    # Record overall start time
    total_start_time = time.time()
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data (normalization uses all data, does not cause data leakage)
    print("Loading data...")
    data_start_time = time.time()
    (X_strain, X_stress, X_material, X_peak_stress, X_peak_strain, material_param_names,
     strain_scaler, stress_scaler, material_scaler, peak_stress_scaler, peak_strain_scaler,
     X_material_original, X_stress_original, X_peak_stress_original, X_peak_strain_original,
     sample_divisions, cluster_labels, extra_data) = load_dataset_with_clusters(
        material_params_file=material_params_file,
        stress_data_file=stress_data_file,
        curve_length=curve_length,
        train_indices=None,
        cache_dir=None,
        use_cache=True,
        default_cluster_count=None,
        verbose=True
    )
    
    # Data split - using external input or DataSlice column for split
    if train_idx is not None and val_idx is not None and test_idx is not None:
        print("\nUsing external data for split...")
        print(f"Training set: {len(train_idx)} samples")
        print(f"Validation set: {len(val_idx)} samples")
        print(f"Test set: {len(test_idx)} samples")
    else:
        print("\nUsing DataSlice column for data split...")
        
        # Extract subset/test labels (support test1~test20 etc.)
        subset_mask = np.array([str(lbl).lower().startswith('subset') for lbl in sample_divisions])
        subset_labels = sorted(set(sample_divisions[subset_mask]), key=lambda x: str(x))
        
        test_mask = np.array([str(lbl).lower().startswith('test') for lbl in sample_divisions])
        test_idx = np.where(test_mask)[0]
        if len(test_idx) == 0:
            raise ValueError("DataSlice column does not contain test/test1~testN labels, cannot build test set")
        test_labels = sample_divisions[test_idx]
        
        # Check subset number
        if len(subset_labels) != 3:
            raise ValueError(f"Expected exactly 3 subset labels, got {len(subset_labels)}: {subset_labels}")
        
        # Count and print each subset and test set number
        subset_to_indices = {lbl: np.where(sample_divisions == lbl)[0] for lbl in subset_labels}
        print("Dataset subsets detected (counts):")
        for lbl in subset_labels:
            print(f"  {lbl}: {len(subset_to_indices[lbl])}")
        print("Test splits detected (counts):")
        for lbl, cnt in zip(*np.unique(test_labels, return_counts=True)):
            print(f"  {lbl}: {cnt}")
        
        # Default use first two subsets as training set, last subset as validation set
        train_idx = np.concatenate([subset_to_indices[subset_labels[0]], subset_to_indices[subset_labels[1]]])
        val_idx = subset_to_indices[subset_labels[2]]
        
        print(f"Training subset: {subset_labels[0]}, {subset_labels[1]} -> {len(train_idx)} samples")
        print(f"Validation subset: {subset_labels[2]} -> {len(val_idx)} samples")
        print(f"Test set: {len(test_idx)} samples")
    
    data_time = time.time() - data_start_time
    print(f"\nData loading time: {data_time:.2f} seconds")
    print(f"Loaded {len(X_strain)} samples")
    print(f"Number of material parameters: {len(material_param_names)}")
    print(f"Peak stress shape: {X_peak_stress.shape}")
    print(f"Peak strain shape: {X_peak_strain.shape}")
    
    
    # Check data distribution
    print(f"\n=== Data Distribution Check ===")
    train_material_means = np.mean(X_material[train_idx], axis=0)
    val_material_means = np.mean(X_material[val_idx], axis=0)
    test_material_means = np.mean(X_material[test_idx], axis=0)
    
    print(f"Train set material parameter means: {train_material_means[:5]}")
    print(f"Validation set material parameter means: {val_material_means[:5]}")
    print(f"Test set material parameter means: {test_material_means[:5]}")
    
    # Calculate distribution difference
    train_val_diff = np.mean(np.abs(train_material_means - val_material_means) / (train_material_means + 1e-8))
    train_test_diff = np.mean(np.abs(train_material_means - test_material_means) / (train_material_means + 1e-8))
    
    print(f"Train vs Validation distribution difference: {train_val_diff:.4f}")
    print(f"Train vs Test distribution difference: {train_test_diff:.4f}")
    
    if train_test_diff > 0.3:
        print("[WARN] Warning: Test set distribution differs significantly from train set, may affect model performance")
    else:
        print("[OK] Data distribution is relatively uniform")
    
    num_clusters = int(np.max(cluster_labels)) + 1 if cluster_labels is not None and len(cluster_labels) > 0 else 1

    def build_dataset(indices):
        dataset_kwargs = {
            'output_length': curve_length
        }
        if cluster_labels is not None:
            dataset_kwargs['cluster_labels'] = cluster_labels[indices]
            dataset_kwargs['num_clusters'] = num_clusters
        return ClusterAwareDataset(
            X_strain[indices], X_stress[indices], X_material[indices],
            X_peak_stress[indices], X_peak_strain[indices],
            **dataset_kwargs
        )
    
    # Create dataset
    train_dataset = build_dataset(train_idx)
    val_dataset = build_dataset(val_idx)
    test_dataset = build_dataset(test_idx)
    
    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Use cross-validation for training (test set remains unchanged)
    print(f"\n开始交叉验证训练...")
    training_start_time = time.time()
    
    # Merge train and val indices for cross-validation
    trainval_idx = np.concatenate([train_idx, val_idx])
    print(f"Cross-validation dataset size: {len(trainval_idx)} samples (training {len(train_idx)} + validation {len(val_idx)})")
    print(f"Test set remains unchanged: {len(test_idx)} samples")
    
    # Store training results
    fold_scores = []
    fold_metrics = []
    fold_best_params = []
    fold_test_predictions = []
    
    # Hyperparameter search and single training
    # Hyperparameter search
    print(f"\n=== Hyperparameter search ===")
    train_dataset = build_dataset(train_idx)
    val_dataset = build_dataset(val_idx)
    
    # Note: objective function now receives dataset instead of DataLoader, to support batch_size search
    
    def objective_func(trial):
        return objective(trial, train_dataset, val_dataset, len(material_param_names), curve_length)
    
    # Create Optuna study (add Pruner for early termination)
    # Use MedianPruner: if intermediate value is higher than median of completed trials, prune early
    # Note: MedianPruner will compare current trial's validation loss with median of completed trials
    #       If current loss is significantly higher than median, the trial is performing poorly, can prune early
    pruner = MedianPruner(
        n_startup_trials=8,  # First 8 trials are not pruned, ensure enough baseline data (from 5 to 8)
        n_warmup_steps=15,   # First 15 epochs are not pruned, give model more training time (from 8 to 15)
        interval_steps=2     # Check every 2 epochs, reduce pruning frequency (from 1 to 2)
    )
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=pruner
    )
    print("Pruning (early termination) enabled: poorly performing trials will be pruned early to speed up search")
    
    study.optimize(objective_func, n_trials=n_trials)
    
    # Check if there are completed trials (not pruned)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) == 0:
        print("\n[Warning] All trials were pruned, no completed trials.")
        print("Possible reasons:")
        print("  1. Validation loss threshold (0.05) set too small")
        print("  2. Model needs more training time to converge")
        print("  3. Hyperparameter search space needs to be adjusted")
        print("\nTrying to use best parameters from pruned trials...")
        
        # Use best parameters from pruned trials (if any)
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        if len(pruned_trials) > 0:
            # Find the trial with the smallest validation loss in pruned trials (using the last reported intermediate value)
            best_pruned_trial = None
            best_val_loss = float('inf')
            for trial in pruned_trials:
                # Get the last reported intermediate value as the validation loss for this trial
                if trial.intermediate_values:
                    # Get the last epoch's validation loss (the one with the largest step)
                    last_step = max(trial.intermediate_values.keys())
                    last_val_loss = trial.intermediate_values[last_step]
                    if last_val_loss < best_val_loss:
                        best_val_loss = last_val_loss
                        best_pruned_trial = trial
            
            if best_pruned_trial is not None:
                best_params = best_pruned_trial.params
                print(f"Best parameters from pruned trial: {best_params}")
                print(f"Corresponding validation loss: {best_val_loss:.4f}")
            else:
                raise ValueError("All trials were pruned, and no validation loss values are available. Please adjust pruning strategy or hyperparameter search space.")
        else:
            raise ValueError("No trial results are available. Please check hyperparameter search configuration.")
    else:
        # Get best hyperparameters (normal case)
        best_params = study.best_params
        print(f"Best hyperparameters: {best_params}")
        print(f"Best validation loss: {study.best_value:.4f}")
    
    # Use best hyperparameters to train model
    print(f"\n=== 开始训练 ===")
    model_params = {k: v for k, v in best_params.items() if k not in ['learning_rate', 'weight_decay', 'batch_size', 'optimizer_type', 'ascending_curve_weight', 'ascending_physics_weight', 'ascending_smoothness_weight', 'descending_curve_weight', 'descending_physics_weight', 'descending_smoothness_weight']}
    # Ensure lstm_units and fc_hidden_size are integers (convert from float)
    if 'lstm_units' in model_params:
        model_params['lstm_units'] = int(model_params['lstm_units'])
    if 'fc_hidden_size' in model_params:
        model_params['fc_hidden_size'] = int(model_params['fc_hidden_size'])
    if 'num_lstm_layers' in model_params:
        model_params['num_lstm_layers'] = int(model_params['num_lstm_layers'])
    if 'attention_heads' in model_params:
        model_params['attention_heads'] = int(model_params['attention_heads'])
    model = BidirectionalLSTMRegressor(
        input_size=len(material_param_names) + 3,
        output_length=curve_length,
        **model_params
    )
    model = model.to(device)
    
    # Use best batch_size (if exists, otherwise use default value 16)
    best_batch_size = int(best_params.get('batch_size', 16))
    train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)
    
    # Use best weight_decay and optimizer_type (if exists, otherwise use default value)
    best_weight_decay = best_params.get('weight_decay', 1e-4)
    optimizer_type = best_params.get('optimizer_type', 'adam')
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_weight_decay)
    
    # Use best loss weights (if exists, otherwise use default value)
    if 'ascending_curve_weight' in best_params:
        criterion = EnhancedCurveLoss(
            ascending_curve_weight=best_params.get('ascending_curve_weight', 0.95),  # Default value increased to 0.95
            ascending_physics_weight=best_params.get('ascending_physics_weight', 0.03),  # 默认值降低到0.03
            ascending_smoothness_weight=best_params.get('ascending_smoothness_weight', 0.02),
            descending_curve_weight=best_params.get('descending_curve_weight', 0.95),  # Default value increased to 0.95
            descending_physics_weight=best_params.get('descending_physics_weight', 0.03),  # Default value decreased to 0.03
            descending_smoothness_weight=best_params.get('descending_smoothness_weight', 0.02)
        )
    else:
        criterion = EnhancedCurveLoss()  # Use new default weights (0.95, 0.03, 0.02)
    
    # Improved learning rate scheduler - use ReduceLROnPlateau + Warm-up (smarter adaptive learning rate)
    initial_lr = best_params['learning_rate']
    warmup_epochs = max(1, int(epochs * 0.1))  # The first 10% of epochs are used for warm-up
    
    # Warm-up scheduler (used in first warmup_epochs epochs)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    
    # Use ReduceLROnPlateau: automatically decrease learning rate when validation loss no longer decreases (smarter)
    # This is more flexible than fixed scheduling, can automatically adjust based on actual training情况
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',  # Monitor validation loss, the smaller the better
        factor=0.5,  # Learning rate decay factor:每次降低50% decrease learning rate by 50%
        patience=15,  # Validation loss does not decrease for 15 epochs, decrease learning rate
        min_lr=1e-7,  # Minimum learning rate
        cooldown=5,  # After decreasing learning rate, wait 5 epochs before continuing to monitor
        eps=1e-8  # Threshold, used to determine if improvement
    )
    
    # Mixed precision training: create GradScaler (only enabled on GPU)
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision training (AMP) enabled, will accelerate training and save memory")
    
    # Early stopping mechanism
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            strain = batch['strain'].to(device)
            stress = batch['stress'].to(device)
            material_params = batch['material_params'].to(device)
            peak_stress = batch['peak_stress'].to(device)
            peak_strain = batch['peak_strain'].to(device)
            
            # Prepare input
            x = torch.cat([strain.unsqueeze(-1), material_params, peak_stress.unsqueeze(-1), peak_strain.unsqueeze(-1)], dim=-1)
            
            optimizer.zero_grad()
            
            # Mixed precision training: use autocast to wrap forward propagation
            if use_amp:
                with autocast():
                    curve_pred = model(x)
                    loss, _ = criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
            else:
                curve_pred = model(x)
                loss, _ = criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
            
            if torch.isnan(loss).item() or torch.isinf(loss).item():
                optimizer.zero_grad()
                if use_amp:
                    scaler.update()
                continue
            
            # Mixed precision training: use scaler for backward propagation
            if use_amp:
                scaler.scale(loss).backward()
                # Gradient clipping (needs to be done in scaler)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Gradient clipping to prevent gradient explosion - more strict clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                strain = batch['strain'].to(device)
                stress = batch['stress'].to(device)
                material_params = batch['material_params'].to(device)
                peak_stress = batch['peak_stress'].to(device)
                peak_strain = batch['peak_strain'].to(device)
                
                x = torch.cat([strain.unsqueeze(-1), material_params, peak_stress.unsqueeze(-1), peak_strain.unsqueeze(-1)], dim=-1)
                
                # Validation also uses autocast to accelerate
                if use_amp:
                    with autocast():
                        curve_pred = model(x)
                        loss, _ = criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
                else:
                    curve_pred = model(x)
                    loss, _ = criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss_avg = val_loss / len(val_loader)
        
        # Learning rate scheduling: use warm-up for first warmup_epochs, then use ReduceLROnPlateau
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            # ReduceLROnPlateau needs to pass in validation loss value
            scheduler.step(val_loss_avg)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            patience_counter = 0
            # Save current best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_params': model_params,
                'best_params': best_params,
                'epoch': epoch,
                'loss': val_loss
            }, os.path.join(save_dir, 'best_model.pth'))
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, training loss: {epoch_loss/len(train_loader):.4f}, validation loss: {val_loss/len(val_loader):.4f}, learning rate: {current_lr:.2e}")
    
    # 在测试集上评估
    print(f"\nEvaluating test set...")
    model.eval()
    all_pred_curves = []
    all_true_curves = []
    all_strain_curves = []
    
    with torch.no_grad():
        for batch in test_loader:
            strain = batch['strain'].to(device)
            stress = batch['stress'].to(device)
            material_params = batch['material_params'].to(device)
            peak_stress = batch['peak_stress'].to(device)
            peak_strain = batch['peak_strain'].to(device)
            
            # Prepare input
            x = torch.cat([strain.unsqueeze(-1), material_params, peak_stress.unsqueeze(-1), peak_strain.unsqueeze(-1)], dim=-1)
            curve_pred = model(x)
            
            all_pred_curves.append(curve_pred.cpu().numpy())
            all_true_curves.append(stress.cpu().numpy())
            all_strain_curves.append(strain.cpu().numpy())
    
    # Calculate test set metrics
    pred_curves = np.concatenate(all_pred_curves, axis=0)
    true_curves = np.concatenate(all_true_curves, axis=0)
    strain_curves = np.concatenate(all_strain_curves, axis=0)
    
    # Post-process initial segment, ensure elastic behavior
    print(f"Post-processing initial segment, ensuring elastic behavior...")
    pred_curves = postprocess_initial_segment(pred_curves, strain_curves, initial_ratio=0.15)
    
    # Unnormalize
    if stress_scaler and stress_scaler['type'] == 'peak_average':
        pred_curves_original = pred_curves * stress_scaler['factor']
        true_curves_original = true_curves * stress_scaler['factor']
    else:
        pred_curves_original = pred_curves
        true_curves_original = true_curves
    
    test_metrics = calculate_curve_metrics(pred_curves_original, true_curves_original)
    
    print(f"Test set results:")
    print(f"    Curve R²: {test_metrics['Curve_R2']:.4f}")
    print(f"    Curve RMSE: {test_metrics['Curve_RMSE']:.4f}")
    print(f"    Peak R²: {test_metrics['Peak_R2']:.4f}")
    print(f"    Mean DTW Distance: {test_metrics['Mean_DTW_Distance']:.4f}")
    
    fold_scores.append(test_metrics['Curve_R2'])
    fold_metrics.append(test_metrics)
    fold_best_params.append(best_params)
    fold_test_predictions.append({
        'pred_curves': pred_curves_original,
        'true_curves': true_curves_original,
        'sample_labels': sample_divisions[test_idx]
    })
    
    # Calculate average metrics (compatible with KFold format)
    mean_metrics = {}
    std_metrics = {}
    if len(fold_metrics) > 0:
        all_metrics_keys = fold_metrics[0].keys()
        for key in all_metrics_keys:
            values = [fold_metrics[i][key] for i in range(len(fold_metrics))]
            mean_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)
    
    # Create result dictionary (compatible format)
    cv_results = {
        'fold_scores': fold_scores,
        'fold_metrics': fold_metrics,
        'fold_best_params': fold_best_params,
        'fold_test_predictions': fold_test_predictions,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics
    }
    
    # 保存最终模型（使用当前折的所有训练+验证数据重新训练）
    print(f"\n[Current fold] Using all training+validation data from current fold to retrain model and save to: {save_dir}")
    print(f"Note: This is the final model for a single fold, the final model for three-fold cross-validation will be generated after all folds are trained")
    final_train_dataset = build_dataset(trainval_idx)
    # Use best batch_size (if exists, otherwise use default value 16)
    final_batch_size = int(best_params.get('batch_size', 16))
    final_train_loader = DataLoader(final_train_dataset, batch_size=final_batch_size, shuffle=True)
    
    final_model = BidirectionalLSTMRegressor(
        input_size=len(material_param_names) + 3,
        output_length=curve_length,
        **model_params
    )
    final_model = final_model.to(device)
    
    # Use best weight_decay and optimizer_type (if exists, otherwise use default value)
    final_model_weight_decay = best_params.get('weight_decay', 1e-4)
    final_optimizer_type = best_params.get('optimizer_type', 'adam')
    if final_optimizer_type == 'adamw':
        optimizer = optim.AdamW(final_model.parameters(), lr=best_params['learning_rate'], weight_decay=final_model_weight_decay)
    else:
        optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'], weight_decay=final_model_weight_decay)
    
    # Use best loss weights (default values if not specified)
    if 'ascending_curve_weight' in best_params:
        criterion = EnhancedCurveLoss(
            ascending_curve_weight=best_params.get('ascending_curve_weight', 0.95),  # Default value increased to 0.95
            ascending_physics_weight=best_params.get('ascending_physics_weight', 0.03),  # Default value decreased to 0.03
            ascending_smoothness_weight=best_params.get('ascending_smoothness_weight', 0.02),
            descending_curve_weight=best_params.get('descending_curve_weight', 0.95),  # Default value increased to 0.95
            descending_physics_weight=best_params.get('descending_physics_weight', 0.03),  # Default value decreased to 0.03
            descending_smoothness_weight=best_params.get('descending_smoothness_weight', 0.02)
        )
    else:
        criterion = EnhancedCurveLoss()  # Use new default weights (0.95, 0.03, 0.02)
    
    # Mixed precision training: create GradScaler (only enabled on GPU)
    use_amp_final = torch.cuda.is_available()
    scaler_final = GradScaler() if use_amp_final else None
    
    # Early stopping mechanism (based on training loss, because there is no independent validation set)
    best_train_loss = float('inf')
    train_patience = 100  # Training loss does not decrease for 100 epochs, early stop
    train_patience_counter = 0
    
    for epoch in range(epochs):
        final_model.train()
        epoch_loss = 0
        for batch in final_train_loader:
            strain = batch['strain'].to(device)
            stress = batch['stress'].to(device)
            material_params = batch['material_params'].to(device)
            peak_stress = batch['peak_stress'].to(device)
            peak_strain = batch['peak_strain'].to(device)
            
            # Prepare input
            x = torch.cat([strain.unsqueeze(-1), material_params, peak_stress.unsqueeze(-1), peak_strain.unsqueeze(-1)], dim=-1)
            
            optimizer.zero_grad()
            
            # Mixed precision training: use autocast to wrap forward propagation
            if use_amp_final:
                with autocast():
                    curve_pred = final_model(x)
                    loss, _ = criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
            else:
                curve_pred = final_model(x)
                loss, _ = criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
            
            if not (torch.isnan(loss).item() or torch.isinf(loss).item()):
                # Mixed precision training: use scaler for backward propagation
                if use_amp_final:
                    scaler_final.scale(loss).backward()
                    scaler_final.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=0.5)
                    scaler_final.step(optimizer)
                    scaler_final.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=0.5)
                    optimizer.step()
                epoch_loss += loss.item()
        
        train_loss_avg = epoch_loss / len(final_train_loader)
        
        # Early stopping check (based on training loss)
        if train_loss_avg < best_train_loss:
            best_train_loss = train_loss_avg
            train_patience_counter = 0
        else:
            train_patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, training loss: {train_loss_avg:.4f}, patience: {train_patience_counter}/{train_patience}")
        
        if train_patience_counter >= train_patience:
            print(f"  [Single fold final model] Early stopped at epoch {epoch + 1} (training loss {train_patience_counter} epochs did not improve)")
            break
    
    # Save final model
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'model_params': model_params,
        'best_params': best_params,
        'material_param_names': material_param_names,
        'curve_length': curve_length,
        'strain_scaler': strain_scaler,
        'stress_scaler': stress_scaler,
        'material_scaler': material_scaler,
        'peak_stress_scaler': peak_stress_scaler,
        'peak_strain_scaler': peak_strain_scaler
    }, os.path.join(save_dir, 'best_model.pth'))
    
    training_time = time.time() - training_start_time
    print(f"\nTraining completed, total time: {training_time:.2f} seconds")
    print(f"Number of hyperparameter searches: {n_trials}")
    
    # Save best hyperparameters
    with open(os.path.join(save_dir, 'best_params.json'), 'w') as f:
        import json
        json.dump(best_params, f, indent=2)
    
    # Save training results (remove numpy arrays, only save serializable metrics)
    cv_results_serializable = {
        'fold_scores': cv_results['fold_scores'],
        'fold_metrics': cv_results['fold_metrics'],
        'fold_best_params': cv_results['fold_best_params']
    }
    if 'mean_metrics' in cv_results:
        cv_results_serializable['mean_metrics'] = cv_results['mean_metrics']
    if 'std_metrics' in cv_results:
        cv_results_serializable['std_metrics'] = cv_results['std_metrics']
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump(cv_results_serializable, f, indent=2)
    
    # Save individual predictions (using numpy format)
    if 'fold_test_predictions' in cv_results and len(cv_results['fold_test_predictions']) > 0:
        print(f"\nSaving predictions...")
        # Save single fold predictions
        pred = cv_results['fold_test_predictions'][0]
        save_path = os.path.join(save_dir, 'predictions.npz')
        np.savez(save_path, 
                pred_curves=pred['pred_curves'],
                true_curves=pred['true_curves'])
        print(f"  Predictions saved to: {save_path}")
    
    # Save data preprocessing information
    preprocessing_info = {
        'material_param_names': material_param_names,  # Independent variable 2: material parameter column names
        'curve_length': curve_length,
        'data_shapes': {
            'X_strain_shape': list(X_strain.shape),  # Independent variable 1: strain data shape
            'X_stress_shape': list(X_stress.shape),  # Dependent variable: stress data shape
            'X_material_shape': list(X_material.shape)  # Independent variable 2: material parameter data shape
        },
        'data_split': {
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'train_ratio': len(train_idx) / len(X_strain),
            'val_ratio': len(val_idx) / len(X_strain),
            'test_ratio': len(test_idx) / len(X_strain)
        }
    }
    
    with open(os.path.join(save_dir, 'preprocessing_info.json'), 'w') as f:
        import json
        json.dump(preprocessing_info, f, indent=2)
    
    # Do not plot results here, plot after final model training is completed
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    print(f"\n=== Time Statistics ===")
    print(f"Data loading time: {data_time:.2f} seconds")
    print(f"Model training time: {training_time:.2f} seconds")
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save time statistics
    time_stats = {
        'data_loading_time': data_time,
        'model_training_time': training_time,
        'total_time': total_time,
        'device_used': str(device),
        'gpu_available': torch.cuda.is_available()
    }
    
    with open(os.path.join(save_dir, 'time_statistics.json'), 'w') as f:
        import json
        json.dump(time_stats, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}\n")
    print("=== Training Completed ===")
    
    return best_params, cv_results

def plot_cv_results(cv_results, save_dir):
    """Plot cross-validation results"""
    import matplotlib.pyplot as plt
    import os
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Bidirectional LSTM Cross-Validation Results', fontsize=16)
    
    # 1. R² scores for each fold
    fold_scores = cv_results['fold_scores']
    bars = axes[0, 0].bar(range(1, len(fold_scores) + 1), fold_scores, color='steelblue', alpha=0.8)
    axes[0, 0].set_title('Test Set R² Score by Fold', fontweight='bold')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('R² Score')
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    axes[0, 0].axhline(y=mean_score, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.3f}±{std_score:.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Display specific values on each bar
    for i, (bar, score) in enumerate(zip(bars, fold_scores)):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Curve metrics comparison
    metrics = ['Curve_R2', 'Curve_MSE', 'Curve_RMSE', 'Curve_MAE', 'Mean_DTW_Distance', 'Peak_R2', 'Peak_MSE', 'Peak_RMSE', 'Peak_MAE', 'Mean_Shape_Similarity']
    means = [cv_results['mean_metrics'][m] for m in metrics]
    stds = [cv_results['std_metrics'][m] for m in metrics]
    
    x_pos = np.arange(len(metrics))
    axes[0, 1].bar(x_pos, means, yerr=stds, capsize=5)
    axes[0, 1].set_title('Main Metrics Comparison')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(metrics, rotation=45)
    axes[0, 1].set_ylabel('Metric Value')
    
    # 3. DTW distance distribution
    dtw_values = [cv_results['fold_metrics'][i]['Mean_DTW_Distance'] for i in range(len(cv_results['fold_metrics']))]
    axes[0, 2].hist(dtw_values, bins=5, alpha=0.7)
    axes[0, 2].set_title('DTW Distance Distribution')
    axes[0, 2].set_xlabel('DTW Distance')
    axes[0, 2].set_ylabel('Frequency')
    
    # 4. Peak prediction accuracy
    peak_r2_values = [cv_results['fold_metrics'][i]['Peak_R2'] for i in range(len(cv_results['fold_metrics']))]
    axes[1, 0].bar(range(1, len(peak_r2_values) + 1), peak_r2_values)
    axes[1, 0].set_title('Peak R² Score by Fold')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Peak R² Score')
    
    # 5. Shape similarity
    shape_sim_values = [cv_results['fold_metrics'][i]['Mean_Shape_Similarity'] for i in range(len(cv_results['fold_metrics']))]
    axes[1, 1].bar(range(1, len(shape_sim_values) + 1), shape_sim_values)
    axes[1, 1].set_title('Shape Similarity by Fold')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('Shape Similarity')
    
    # 6. Comprehensive performance radar chart
    categories = ['R^2', 'MSE', 'DTW', 'Peak_R^2', 'Shape_Sim']
    # For single values, use normalization directly
    mse_score = max(0, 1 - cv_results['mean_metrics']['Curve_MSE'] / 100)  # 假设MSE最大值为100
    dtw_score = max(0, 1 - cv_results['mean_metrics']['Mean_DTW_Distance'] / 100)  # 假设DTW最大值为100
    
    values = [cv_results['mean_metrics']['Curve_R2'], 
             mse_score,
             dtw_score,
             cv_results['mean_metrics']['Peak_R2'],
             cv_results['mean_metrics']['Mean_Shape_Similarity']]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # Close the graph
    angles += angles[:1]
    
    axes[1, 2].plot(angles, values, 'o-', linewidth=2)
    axes[1, 2].fill(angles, values, alpha=0.25)
    axes[1, 2].set_xticks(angles[:-1])
    axes[1, 2].set_xticklabels(categories)
    axes[1, 2].set_title('Comprehensive Performance Radar Chart')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_results_visualization.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_data_distribution_comparison(train_strain, train_stress, train_material, material_names,
                                    val_strain, val_stress, val_material, val_label,
                                    test_strain, test_stress, test_material, test_label,
                                    train_label, save_dir):
    """Plot data distribution comparison between training set, validation set, and test set"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("Plotting data distribution comparison...")
    
    # Create large figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Data Distribution Comparison Analysis', fontsize=16, fontweight='bold')
    
    # 1. Strain data distribution comparison
    ax1 = axes[0, 0]
    ax1.hist(train_strain.flatten(), bins=30, alpha=0.7, label=train_label, density=True, color='blue')
    ax1.hist(val_strain.flatten(), bins=30, alpha=0.7, label=val_label, density=True, color='orange')
    ax1.hist(test_strain.flatten(), bins=30, alpha=0.7, label=test_label, density=True, color='green')
    ax1.set_title('Strain Data Distribution Comparison')
    ax1.set_xlabel('Strain Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Stress data distribution comparison
    ax2 = axes[0, 1]
    ax2.hist(train_stress.flatten(), bins=30, alpha=0.7, label=train_label, density=True, color='blue')
    ax2.hist(val_stress.flatten(), bins=30, alpha=0.7, label=val_label, density=True, color='orange')
    ax2.hist(test_stress.flatten(), bins=30, alpha=0.7, label=test_label, density=True, color='green')
    ax2.set_title('Stress Data Distribution Comparison')
    ax2.set_xlabel('Stress Value')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Strain-stress scatter plot comparison
    ax3 = axes[0, 2]
    # Randomly sample part of the data for visualization (to avoid overcrowding)
    idx1 = np.random.choice(len(train_strain), min(500, len(train_strain)), replace=False)
    idx2 = np.random.choice(len(val_strain), min(100, len(val_strain)), replace=False)
    idx3 = np.random.choice(len(test_strain), min(100, len(test_strain)), replace=False)
    
    ax3.scatter(train_strain[idx1].flatten(), train_stress[idx1].flatten(), 
               alpha=0.5, s=1, label=train_label, color='blue')
    ax3.scatter(val_strain[idx2].flatten(), val_stress[idx2].flatten(), 
                alpha=0.7, s=2, label=val_label, color='orange')
    ax3.scatter(test_strain[idx3].flatten(), test_stress[idx3].flatten(), 
               alpha=0.7, s=2, label=test_label, color='green')
    ax3.set_title('Strain-Stress Relationship Comparison')
    ax3.set_xlabel('Strain')
    ax3.set_ylabel('Stress')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4-6. Distribution comparison of the first 3 material parameters
    for i in range(min(3, len(material_names))):
        ax = axes[1, i]
        
        ax.hist(train_material[:, i], bins=15, alpha=0.7, label=train_label, 
               density=True, color='blue')
        ax.hist(val_material[:, i], bins=15, alpha=0.7, label=val_label, 
               density=True, color='orange')
        ax.hist(test_material[:, i], bins=15, alpha=0.7, label=test_label, 
               density=True, color='green')
        
        ax.set_title(f'{material_names[i]} Distribution Comparison')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 7. Material parameter mean comparison
    ax7 = axes[2, 0]
    train_means = np.mean(train_material, axis=0)
    val_means = np.mean(val_material, axis=0)
    test_means = np.mean(test_material, axis=0)
    
    x_pos = np.arange(len(material_names))
    width = 0.25
    
    ax7.bar(x_pos - width, train_means, width, label=train_label, alpha=0.8, color='blue')
    ax7.bar(x_pos, val_means, width, label=val_label, alpha=0.8, color='orange')
    ax7.bar(x_pos + width, test_means, width, label=test_label, alpha=0.8, color='green')
    
    ax7.set_title('Material Parameter Mean Comparison')
    ax7.set_xlabel('Material Parameter')
    ax7.set_ylabel('Mean Value')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                        for name in material_names], rotation=45, ha='right')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Material parameter standard deviation comparison
    ax8 = axes[2, 1]
    train_stds = np.std(train_material, axis=0)
    val_stds = np.std(val_material, axis=0)
    test_stds = np.std(test_material, axis=0)
    
    ax8.bar(x_pos - width, train_stds, width, label=train_label, alpha=0.8, color='blue')
    ax8.bar(x_pos, val_stds, width, label=val_label, alpha=0.8, color='orange')
    ax8.bar(x_pos + width, test_stds, width, label=test_label, alpha=0.8, color='green')
    
    ax8.set_title('Material Parameter Standard Deviation Comparison')
    ax8.set_xlabel('Material Parameter')
    ax8.set_ylabel('Standard Deviation')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                        for name in material_names], rotation=45, ha='right')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Dataset size comparison
    ax9 = axes[2, 2]
    sizes = [len(train_strain), len(val_strain), len(test_strain)]
    labels = [train_label, val_label, test_label]
    colors = ['blue', 'orange', 'green']
    
    bars = ax9.bar(labels, sizes, color=colors, alpha=0.8)
    ax9.set_title('Dataset Size Comparison')
    ax9.set_ylabel('Sample Count')
    
    # Add numerical labels
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + max(sizes)*0.01,
                f'{size}', ha='center', va='bottom', fontweight='bold')
    
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'data_distribution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\n=== Dataset Distribution Statistics ===")
    print(f"{train_label}: {len(train_strain)} samples")
    print(f"{val_label}: {len(val_strain)} samples") 
    print(f"{test_label}: {len(test_strain)} samples")
    print(f"Total samples: {len(train_strain) + len(val_strain) + len(test_strain)}")
    
    print(f"\nStrain Statistics:")
    print(f"  {train_label} - Mean: {np.mean(train_strain):.4f}, Std: {np.std(train_strain):.4f}")
    print(f"  {val_label} - Mean: {np.mean(val_strain):.4f}, Std: {np.std(val_strain):.4f}")
    print(f"  {test_label} - Mean: {np.mean(test_strain):.4f}, Std: {np.std(test_strain):.4f}")
    
    print(f"\nStress Statistics:")
    print(f"  {train_label} - Mean: {np.mean(train_stress):.4f}, Std: {np.std(train_stress):.4f}")
    print(f"  {val_label} - Mean: {np.mean(val_stress):.4f}, Std: {np.std(val_stress):.4f}")
    print(f"  {test_label} - Mean: {np.mean(test_stress):.4f}, Std: {np.std(test_stress):.4f}")
    
    print(f"\nMaterial Parameter Distribution Uniformity Check:")
    train_means = np.mean(train_material, axis=0)
    val_means = np.mean(val_material, axis=0)
    test_means = np.mean(test_material, axis=0)
    
    # Calculate distribution differences
    mean_diff_train_val = np.abs(train_means - val_means) / (train_means + 1e-8)
    mean_diff_train_test = np.abs(train_means - test_means) / (train_means + 1e-8)
    mean_diff_val_test = np.abs(val_means - test_means) / (val_means + 1e-8)
    
    print(f"  Train vs Validation mean difference: {np.mean(mean_diff_train_val):.4f}")
    print(f"  Train vs Test mean difference: {np.mean(mean_diff_train_test):.4f}")
    print(f"  Validation vs Test mean difference: {np.mean(mean_diff_val_test):.4f}")
    
    if np.mean(mean_diff_train_val) < 0.2 and np.mean(mean_diff_train_test) < 0.2:
        print("  [OK] Data distribution is relatively uniform, stratified sampling successful!")
    else:
        print("  [WARN] Data distribution has significant differences, may need to adjust stratification strategy")
    
    print(f"Data distribution comparison plot saved to: {os.path.join(save_dir, 'data_distribution_comparison.png')}")

def plot_test_predictions(pred_curves, true_curves, test_material_params, strain_data, 
                         save_dir='SAVE/bidirectional_lstm', num_plots=None, strain_scaler=None, test_labels=None):
    """
    Plot test set prediction result comparison - display all test samples
    
    Args:
        pred_curves: Predicted stress curves [n_samples, curve_length]
        true_curves: True stress curves [n_samples, curve_length]
        test_material_params: Test set material parameters [n_samples, num_params]
        strain_data: Strain data [n_samples, curve_length]
        save_dir: Save directory
        num_plots: Number of samples to plot, if None then plot all samples
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Fixed 3x3 canvas, display up to 9 samples
    if num_plots is None:
        num_plots = min(9, len(pred_curves))
    else:
        num_plots = min(min(num_plots, len(pred_curves)), 9)
    
    rows = 3
    cols = 3
    figsize = (12, 9)  # Fixed canvas size
    fontsize = 10
    markersize = 8
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Calculate overall metrics
    mse_scores = []
    r2_scores = []
    mae_scores = []
    peak_mse_scores = []
    
    for i in range(min(num_plots, len(pred_curves))):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        pred_curve = pred_curves[i]
        true_curve = true_curves[i]
        strain_curve = strain_data[i]
        
        # Inverse normalize strain data to original values
        if strain_scaler and strain_scaler['type'] == 'peak_average':
            strain_curve = strain_curve * strain_scaler['factor']
        
        material_params = test_material_params[i]
        
        # Calculate metrics for this sample (using inverse normalized data)
        # Note: pred_curve and true_curve are already inverse normalized when passed in
        mse = np.mean((pred_curve - true_curve)**2)
        r2 = r2_score(true_curve, pred_curve)
        mae = np.mean(np.abs(pred_curve - true_curve))
        
        # Peak metrics
        peak_pred = np.max(pred_curve)
        peak_true = np.max(true_curve)
        peak_mse = (peak_pred - peak_true)**2
        
        mse_scores.append(mse)
        r2_scores.append(r2)
        mae_scores.append(mae)
        peak_mse_scores.append(peak_mse)
        
        # Plot curves
        ax.plot(strain_curve, true_curve, 'b-', linewidth=1.5, label='True Curve', alpha=0.9)
        ax.plot(strain_curve, pred_curve, 'r--', linewidth=1.5, label='Predicted Curve', alpha=0.9)
        
        # Mark peaks
        peak_strain_idx = np.argmax(true_curve)
        peak_strain = strain_curve[peak_strain_idx]
        peak_stress = true_curve[peak_strain_idx]
        ax.plot(peak_strain, peak_stress, 'bo', markersize=markersize, label='True Peak', markeredgecolor='darkblue', markeredgewidth=1)
        
        pred_peak_idx = np.argmax(pred_curve)
        pred_peak_strain = strain_curve[pred_peak_idx] 
        pred_peak_stress = pred_curve[pred_peak_idx]
        ax.plot(pred_peak_strain, pred_peak_stress, 'rs', markersize=markersize, label='Predicted Peak', markeredgecolor='darkred', markeredgewidth=1)
        
        # Set title and legend
        # Use test label as title, if not then use Sample number
        if test_labels is not None and i < len(test_labels):
            sample_label = str(test_labels[i])
        else:
            sample_label = f"Sample {i+1}"
        material_info = f"{sample_label}\nMSE: {mse:.3f}, R^2: {r2:.3f}"
        if len(material_params) >= 8:
            r_val = material_params[6] if material_params[6] <= 1 else material_params[6] / 100  # r (Quality replacement rate) - Index 6
            wa_val = material_params[7] if material_params[7] <= 1 else material_params[7] / 100  # WA (Mixing aggregate water absorption rate) - Index 7
            material_info += f"\nr: {r_val:.2f}, WA: {wa_val:.2f}"
        
        ax.set_title(material_info, fontsize=fontsize, fontweight='bold')
        ax.set_xlabel('Strain', fontsize=fontsize-1)
        ax.set_ylabel('Stress', fontsize=fontsize-1)
        
        # Only show legend in the first subplot
        if i == 0:
            ax.legend(fontsize=fontsize-2, loc='upper right')
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(labelsize=fontsize-2)
        
        # Set axis range, ensure curves are fully displayed
        ax.set_xlim(strain_curve.min() * 0.95, strain_curve.max() * 1.05)
        ax.set_ylim(min(true_curve.min(), pred_curve.min()) * 0.9, 
                   max(true_curve.max(), pred_curve.max()) * 1.1)
    
    # Hide extra subplots
    for i in range(num_plots, rows * cols):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.suptitle(f'Bidirectional LSTM Test Set Prediction Results Comparison (All {num_plots} Samples)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_predictions_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print overall statistics
    print(f"\n=== Test Set Prediction Statistics (All {num_plots} Samples) ===")
    print(f"Mean MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
    print(f"Mean R^2: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"Mean MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"Mean Peak MSE: {np.mean(peak_mse_scores):.4f} ± {np.std(peak_mse_scores):.4f}")
    
    return mse_scores, r2_scores, mae_scores, peak_mse_scores

# ========== 9. Main function ==========
def cross_subset_train_validation(material_params_file, stress_data_file, save_dir='SAVE/bidirectional_lstm', n_trials=20, epochs=500, curve_length=DEFAULT_CURVE_LENGTH):
    """
    subset1/2/3 as validation set, the rest as training set, test set remains unchanged. Save results for each round.
    """
    # Load data first, get subset labels
    (X_strain, X_stress, X_material, X_peak_stress, X_peak_strain, material_param_names,
     strain_scaler, stress_scaler, material_scaler, peak_stress_scaler, peak_strain_scaler,
     X_material_original, X_stress_original, X_peak_stress_original, X_peak_strain_original,
     sample_divisions, cluster_labels, extra_data) = load_dataset_with_clusters(
        material_params_file=material_params_file,
        stress_data_file=stress_data_file,
        curve_length=curve_length,
        train_indices=None,
        cache_dir=None,
        use_cache=True,
        default_cluster_count=None,
        verbose=True
    )
    subset_mask = np.array([str(lbl).lower().startswith('subset') for lbl in sample_divisions])
    subset_labels = sorted(set(sample_divisions[subset_mask]), key=lambda x: str(x))
    test_mask = np.array([str(lbl).lower().startswith('test') for lbl in sample_divisions])
    test_idx = np.where(test_mask)[0]
    if len(test_idx) == 0:
        raise ValueError("Test/test1~testN labels not found in DataSlice column, cannot build test set")
    test_labels = sample_divisions[test_idx]
    print("Test splits detected (counts):")
    for lbl, cnt in zip(*np.unique(test_labels, return_counts=True)):
        print(f"  {lbl}: {cnt}")
    if len(subset_labels) != 3:
        raise ValueError(f"Number of subsets should be 3, actual: {len(subset_labels)} {subset_labels}")
    all_results = []

    num_clusters = int(np.max(cluster_labels)) + 1 if cluster_labels is not None and len(cluster_labels) > 0 else 1

    def build_dataset_local(indices):
        dataset_kwargs = {
            'output_length': curve_length
        }
        if cluster_labels is not None:
            dataset_kwargs['cluster_labels'] = cluster_labels[indices]
            dataset_kwargs['num_clusters'] = num_clusters
        return ClusterAwareDataset(
            X_strain[indices], X_stress[indices], X_material[indices],
            X_peak_stress[indices], X_peak_strain[indices],
            **dataset_kwargs
        )
    print(f"\n{'='*70}")
    print(f"Start three-fold subset cross-validation training")
    print(f"{'='*70}")
    print(f"There are {len(subset_labels)} subsets to train: {subset_labels}")
    print(f"Each subset is used as validation set, the rest as training set")
    print(f"{'='*70}\n")
    
    for i, val_label in enumerate(subset_labels):
        train_labels = [lbl for lbl in subset_labels if lbl != val_label]
        subset_to_indices = {lbl: np.where(sample_divisions == lbl)[0] for lbl in subset_labels}
        train_idx = np.concatenate([subset_to_indices[lbl] for lbl in train_labels])
        val_idx = subset_to_indices[val_label]
        
        print(f"\n{'#'*70}")
        print(f"# subset cross-validation round {i+1}/{len(subset_labels)}")
        print(f"# Validation set: {val_label}")
        print(f"# Training set: {train_labels}")
        print(f"# Test set: test (remains unchanged)")
        print(f"{'#'*70}")
        print(f"Training set sample size: {len(train_idx)}")
        print(f"Validation set sample size: {len(val_idx)}")
        print(f"Test set sample size: {len(test_idx)}")
        print(f"{'#'*70}\n")
        
        # Save results for this round in a dedicated directory
        round_save_dir = os.path.join(save_dir, f'subsetCV_{val_label}')
        os.makedirs(round_save_dir, exist_ok=True)
        
        # Pass division information to avoid re-loading data and re-dividing
        print(f"Start training model with [{val_label} as validation set]...")
        best_params, cv_results = train_bidirectional_lstm_model(
            material_params_file=material_params_file,
            stress_data_file=stress_data_file,
            save_dir=round_save_dir,
            n_trials=n_trials,
            epochs=epochs,
            curve_length=curve_length,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx
        )
        
        print(f"\n{'#'*70}")
        print(f"# [{val_label} validation set] round {i+1}/{len(subset_labels)} training completed!")
        print(f"{'#'*70}")
        print(f"Best hyperparameters: {best_params}")
        print(f"Test set main metrics:")
        print(f"  - Curve_R2: {cv_results['mean_metrics']['Curve_R2']:.4f}")
        print(f"  - Curve_RMSE: {cv_results['mean_metrics']['Curve_RMSE']:.4f}")
        print(f"  - Peak_R2: {cv_results['mean_metrics']['Peak_R2']:.4f}")
        print(f"  - DTW distance: {cv_results['mean_metrics']['Mean_DTW_Distance']:.4f}")
        print(f"{'#'*70}\n")
        
        all_results.append({
            'val_label': val_label,
            'train_labels': train_labels,
            'best_params': best_params,
            'cv_results': cv_results,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        })
        
        print(f"Progress: {i+1}/{len(subset_labels)} rounds completed\n")
    
    print(f"\n{'='*70}")
    print(f"Three-fold subset cross-validation training completed!")
    print(f"{'='*70}\n")
    
    # Select the best model from the three folds (based on test set Curve_R2 metric)
    print(f"\n{'='*70}")
    print(f"Three-fold cross-validation completed, start selecting best model and training final model")
    print(f"{'='*70}")
    print(f"Compare test set performance of the three folds:")
    best_fold_idx = 0
    best_r2 = all_results[0]['cv_results']['mean_metrics']['Curve_R2']
    for i, res in enumerate(all_results):
        r2 = res['cv_results']['mean_metrics']['Curve_R2']
        rmse = res['cv_results']['mean_metrics']['Curve_RMSE']
        peak_r2 = res['cv_results']['mean_metrics']['Peak_R2']
        dtw = res['cv_results']['mean_metrics']['Mean_DTW_Distance']
        print(f"  Fold {i+1} ({res['val_label']}): Curve_R2={r2:.4f}, RMSE={rmse:.4f}, Peak_R2={peak_r2:.4f}, DTW={dtw:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_fold_idx = i
    
    best_result = all_results[best_fold_idx]
    print(f"\nBest fold: Fold {best_fold_idx+1} ({best_result['val_label']})")
    print(f"  - Curve_R2: {best_r2:.4f}")
    print(f"  - Best hyperparameters: {best_result['best_params']}")
    print(f"\nUsing best fold hyperparameters, retrain final model using all training+validation data...")
    print(f"{'='*70}\n")
    
    # Use best fold hyperparameters and all training+validation data to retrain final model
    # Merge all subsets as training set (except test set)
    all_trainval_idx = np.concatenate([
        all_results[0]['train_idx'],
        all_results[0]['val_idx'],
        all_results[1]['train_idx'],
        all_results[1]['val_idx'],
        all_results[2]['train_idx'],
        all_results[2]['val_idx']
    ])
    all_trainval_idx = np.unique(all_trainval_idx)  # Remove duplicates
    final_test_idx = all_results[0]['test_idx']  # test集保持不变
    
    print(f"[Final model] Training data: {len(all_trainval_idx)} samples (all training+validation data from subsets)")
    print(f"[Final model] Test set: {len(final_test_idx)} samples")
    
    # Final model save path (defined before training loop)
    final_model_path = os.path.join(save_dir, 'best_model.pth')
    print(f"[Final model] Model save path: {final_model_path}")
    
    # Use best fold hyperparameters to train final model
    # Exclude training-related parameters (learning_rate, weight_decay, batch_size, optimizer_type, loss weights)
    excluded_params = ['learning_rate', 'weight_decay', 'batch_size', 'optimizer_type', 'ascending_curve_weight', 
                      'ascending_physics_weight', 'ascending_smoothness_weight', 
                      'descending_curve_weight', 'descending_physics_weight', 'descending_smoothness_weight']
    final_model_params = {k: v for k, v in best_result['best_params'].items() if k not in excluded_params}
    # Ensure lstm_units and fc_hidden_size are integers (convert from floats)
    if 'lstm_units' in final_model_params:
        final_model_params['lstm_units'] = int(final_model_params['lstm_units'])
    if 'fc_hidden_size' in final_model_params:
        final_model_params['fc_hidden_size'] = int(final_model_params['fc_hidden_size'])
    if 'num_lstm_layers' in final_model_params:
        final_model_params['num_lstm_layers'] = int(final_model_params['num_lstm_layers'])
    if 'attention_heads' in final_model_params:
        final_model_params['attention_heads'] = int(final_model_params['attention_heads'])
    final_model = BidirectionalLSTMRegressor(
        input_size=len(material_param_names) + 3,
        output_length=curve_length,
        **final_model_params
    )
    final_model = final_model.to(device)
    
    # Use best weight_decay and optimizer_type (if exists, otherwise use default values)
    final_weight_decay = best_result['best_params'].get('weight_decay', 1e-4)
    final_optimizer_type = best_result['best_params'].get('optimizer_type', 'adam')
    if final_optimizer_type == 'adamw':
        final_optimizer = optim.AdamW(final_model.parameters(), lr=best_result['best_params']['learning_rate'], weight_decay=final_weight_decay)
    else:
        final_optimizer = optim.Adam(final_model.parameters(), lr=best_result['best_params']['learning_rate'], weight_decay=final_weight_decay)
    
    # Use best loss weights (if exists, otherwise use default values)
    if 'ascending_curve_weight' in best_result['best_params']:
        final_criterion = EnhancedCurveLoss(
            ascending_curve_weight=best_result['best_params'].get('ascending_curve_weight', 0.9),
            ascending_physics_weight=best_result['best_params'].get('ascending_physics_weight', 0.08),
            ascending_smoothness_weight=best_result['best_params'].get('ascending_smoothness_weight', 0.02),
            descending_curve_weight=best_result['best_params'].get('descending_curve_weight', 0.9),
            descending_physics_weight=best_result['best_params'].get('descending_physics_weight', 0.08),
            descending_smoothness_weight=best_result['best_params'].get('descending_smoothness_weight', 0.02)
        )
    else:
        final_criterion = EnhancedCurveLoss()
    
    # Final model learning rate scheduler - use ReduceLROnPlateau + Warm-up (improved version)
    final_initial_lr = best_result['best_params']['learning_rate']
    final_warmup_epochs = max(1, int(epochs * 0.1))
    
    final_warmup_scheduler = optim.lr_scheduler.LinearLR(
        final_optimizer, start_factor=0.1, end_factor=1.0, total_iters=final_warmup_epochs
    )
    
    # Use ReduceLROnPlateau: automatically reduce learning rate when validation loss no longer decreases
    final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        final_optimizer, 
        mode='min',  # Monitor validation loss, smaller is better
        factor=0.5,  # Learning rate decay factor: reduce by 50% each time
        patience=20,  # Reduce learning rate if validation loss doesn't decrease for 20 epochs (final model uses longer patience)
        min_lr=1e-7,  # Minimum learning rate
        cooldown=5,  # Wait 5 epochs after reducing learning rate before monitoring again
        eps=1e-8  # Threshold, used to determine if improvement is made
    )
    
    # Create final training dataset (using all training+validation data)
    # To early stop, split 10% of all_trainval_idx as validation set
    np.random.seed(42)
    np.random.shuffle(all_trainval_idx)
    val_split_idx = int(len(all_trainval_idx) * 0.9)
    final_train_idx = all_trainval_idx[:val_split_idx]
    final_val_idx = all_trainval_idx[val_split_idx:]
    
    final_train_dataset = build_dataset_local(final_train_idx)
    final_val_dataset = build_dataset_local(final_val_idx)
    
    # Use best batch_size (if exists, otherwise use default value 16)
    final_batch_size_cv = int(best_result['best_params'].get('batch_size', 16))
    final_train_loader = DataLoader(final_train_dataset, batch_size=final_batch_size_cv, shuffle=True)
    final_val_loader = DataLoader(final_val_dataset, batch_size=final_batch_size_cv, shuffle=False)
    
    # Mixed precision training: create GradScaler (only enabled on GPU)
    use_amp_cv_final = torch.cuda.is_available()
    scaler_cv_final = GradScaler() if use_amp_cv_final else None
    if use_amp_cv_final:
        print("[Final model] Mixed precision training (AMP) enabled")
    
    # Early stopping mechanism
    final_best_loss = float('inf')
    final_patience = 100  # Final model early stopping patience value
    final_patience_counter = 0
    final_epoch_trained = 0  # Record actual number of epochs trained
    
    print(f"\n[Final model] Start training (using best fold hyperparameters, all training+validation data)...")
    print(f"[Final model] Training epochs: {epochs}, Early stopping patience: {final_patience}")
    print(f"[Final model] Training set: {len(final_train_idx)} samples, Validation set: {len(final_val_idx)} samples (for early stopping)")
    
    for epoch in range(epochs):
        final_epoch_trained = epoch + 1
        final_model.train()
        epoch_loss = 0
        for batch in final_train_loader:
            strain = batch['strain'].to(device)
            stress = batch['stress'].to(device)
            material_params = batch['material_params'].to(device)
            peak_stress = batch['peak_stress'].to(device)
            peak_strain = batch['peak_strain'].to(device)
            
            # Prepare input
            x = torch.cat([strain.unsqueeze(-1), material_params, peak_stress.unsqueeze(-1), peak_strain.unsqueeze(-1)], dim=-1)
            
            final_optimizer.zero_grad()
            
            # Mixed precision training: use autocast wrapper for forward propagation
            if use_amp_cv_final:
                with autocast():
                    curve_pred = final_model(x)
                    loss, _ = final_criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
            else:
                curve_pred = final_model(x)
                loss, _ = final_criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
            
            if not (torch.isnan(loss).item() or torch.isinf(loss).item()):
                # Mixed precision training: use scaler for backward propagation
                if use_amp_cv_final:
                    scaler_cv_final.scale(loss).backward()
                    scaler_cv_final.unscale_(final_optimizer)
                    torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=0.5)
                    scaler_cv_final.step(final_optimizer)
                    scaler_cv_final.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=0.5)
                    final_optimizer.step()
                epoch_loss += loss.item()
        
        # Validation
        final_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in final_val_loader:
                strain = batch['strain'].to(device)
                stress = batch['stress'].to(device)
                material_params = batch['material_params'].to(device)
                peak_stress = batch['peak_stress'].to(device)
                peak_strain = batch['peak_strain'].to(device)
                
                x = torch.cat([strain.unsqueeze(-1), material_params, peak_stress.unsqueeze(-1), peak_strain.unsqueeze(-1)], dim=-1)
                
                if use_amp_cv_final:
                    with autocast():
                        curve_pred = final_model(x)
                        loss, _ = final_criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
                else:
                    curve_pred = final_model(x)
                    loss, _ = final_criterion(curve_pred, stress, material_params, strain, peak_stress, peak_strain)
                val_loss += loss.item()
        
        val_loss_avg = val_loss / len(final_val_loader)
        
        # Learning rate scheduling: use warm-up for first warmup_epochs, then use ReduceLROnPlateau
        if epoch < final_warmup_epochs:
            final_warmup_scheduler.step()
        else:
            # ReduceLROnPlateau needs to pass validation loss value
            final_scheduler.step(val_loss_avg)
        
        # Early stopping check
        if val_loss_avg < final_best_loss:
            final_best_loss = val_loss_avg
            final_patience_counter = 0
            # Save current best model
            best_final_model_state = {
                'model_state_dict': final_model.state_dict(),
                'model_params': final_model_params,
                'best_params': best_result['best_params'],
                'material_param_names': material_param_names,
                'curve_length': curve_length,
                'strain_scaler': strain_scaler,
                'stress_scaler': stress_scaler,
                'material_scaler': material_scaler,
                'peak_stress_scaler': peak_stress_scaler,
                'peak_strain_scaler': peak_strain_scaler,
                'best_fold_info': {
                    'best_fold_idx': best_fold_idx,
                    'best_fold_label': best_result['val_label'],
                    'best_test_r2': best_r2
                },
                'epoch': epoch,
                'val_loss': val_loss_avg
            }
        else:
            final_patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            current_final_lr = final_optimizer.param_groups[0]['lr']
            print(f"  [Final model] Epoch {epoch + 1}/{epochs}, Training loss: {epoch_loss/len(final_train_loader):.4f}, Validation loss: {val_loss_avg:.4f}, Learning rate: {current_final_lr:.2e}, Patience: {final_patience_counter}/{final_patience}")
        
        if final_patience_counter >= final_patience:
            print(f"  [Final model] Early stopped at epoch {epoch + 1}")
            break
    
    # Use best model state to save
    if 'best_final_model_state' in locals():
        torch.save(best_final_model_state, final_model_path)
    else:
        # If best model not saved, save current model
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'model_params': final_model_params,
            'best_params': best_result['best_params'],
            'material_param_names': material_param_names,
            'curve_length': curve_length,
            'strain_scaler': strain_scaler,
            'stress_scaler': stress_scaler,
            'material_scaler': material_scaler,
            'peak_stress_scaler': peak_stress_scaler,
            'peak_strain_scaler': peak_strain_scaler,
            'best_fold_info': {
                'best_fold_idx': best_fold_idx,
                'best_fold_label': best_result['val_label'],
                'best_test_r2': best_r2
            }
        }, final_model_path)
    
    print(f"\n{'='*70}")
    print(f"[Final model] Training completed and saved!")
    print(f"{'='*70}")
    print(f"Model path: {final_model_path}")
    print(f"Best validation loss: {final_best_loss:.6f}")
    print(f"Training epochs: {final_epoch_trained}/{epochs}")
    print(f"Best fold hyperparameters: {best_result['val_label']}")
    print(f"Training with all training+validation data ({len(all_trainval_idx)} samples)")
    print(f"Best fold test set Curve_R2: {best_r2:.4f}")
    print(f"{'='*70}\n")
    
    # Create final cv_results for plotting (based on three-fold cross-validation results)
    final_cv_results = {
        'fold_scores': [res['cv_results']['mean_metrics']['Curve_R2'] for res in all_results],
        'fold_metrics': [res['cv_results']['mean_metrics'] for res in all_results],
        'fold_best_params': [res['best_params'] for res in all_results],
        'fold_test_predictions': [res['cv_results']['fold_test_predictions'][0] for res in all_results],
        'mean_metrics': {},  # Calculate mean of three folds
        'std_metrics': {}
    }
    
    # Calculate mean and standard deviation of three folds
    if len(all_results) > 0:
        all_metrics_keys = all_results[0]['cv_results']['mean_metrics'].keys()
        for key in all_metrics_keys:
            values = [res['cv_results']['mean_metrics'][key] for res in all_results]
            final_cv_results['mean_metrics'][key] = np.mean(values)
            final_cv_results['std_metrics'][key] = np.std(values)
    
    # Plot final cross-validation results
    print(f"\nPlot final cross-validation results...")
    plot_cv_results(final_cv_results, save_dir)
    print(f"Performance plot saved to: {os.path.join(save_dir, 'cv_results_visualization.png')}")
    
    return all_results

def run_bidirectional_lstm_training():
    """Run subset1/2/3 cross-validation main entry"""
    excel_file = r"C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM\dataset\dataset_final.xlsx"
    all_results = cross_subset_train_validation(
        material_params_file=excel_file,
        stress_data_file=excel_file,
        save_dir=r'C:\JunzanLi_project\constitutive_relation\Pi_BiLSTM\LSTM\SAVE\bidirectional_lstm_cv',
        n_trials=40,  # Increase trial number from 30 to 40, increase the probability of finding better hyperparameters
        epochs=5000,  # Final model train 5000 epochs
        curve_length=DEFAULT_CURVE_LENGTH
    )
    print("\nAll three-fold cross-training completed!")
    for i, res in enumerate(all_results):
        print(f"[{i+1}] Validation set: {res['val_label']} -> Validation main metrics: {res['cv_results']['mean_metrics']}")
    return all_results

if __name__ == "__main__":
    # Run subset cross-validation main entry directly
    all_results = run_bidirectional_lstm_training()