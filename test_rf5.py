import torch
import torch.nn as nn
from timm.models.resnet import resnet18, resnet50, Bottleneck, BasicBlock
from timm.models.alexnet import alexnet
from timm.models.RESMAX import RESMAX_V2_2

# --- Helper function to ensure parameters are tuples ---
def _param_to_tuple(param):
    """Converts an integer or a tuple parameter to a (h, w) tuple."""
    if isinstance(param, int):
        return (param, param)
    return param

# --- Main Analytical RF Calculator Class ---
class RFAnalyzer:
    def __init__(self, enable_upper_bound=True, input_size=224):
        # [height, width]
        self.rf = [1, 1]
        self.j = [1, 1] # Jump (cumulative stride)
        self.results = []
        self.enable_upper_bound = enable_upper_bound
        self.input_size = _param_to_tuple(input_size)
        
    def _update_rf(self, layer_name, layer):
        """Applies the RF formulas for a single conv/pool layer."""
        if not isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            return

        kernel_size = _param_to_tuple(layer.kernel_size)
        stride = _param_to_tuple(layer.stride)
        dilation = _param_to_tuple(getattr(layer, 'dilation', 1))

        # Effective kernel size considering dilation
        # k_eff = k + (k - 1) * (d - 1)
        # k_eff - 1 = (k - 1) * d
        k_eff_h = (kernel_size[0] - 1) * dilation[0]
        k_eff_w = (kernel_size[1] - 1) * dilation[1]
        
        # RF_out = RF_in + (k_eff - 1) * J_in
        self.rf[0] += k_eff_h * self.j[0]
        self.rf[1] += k_eff_w * self.j[1]

        # Apply upper bound if enabled
        if self.enable_upper_bound:
            self.rf[0] = min(self.rf[0], self.input_size[0])
            self.rf[1] = min(self.rf[1], self.input_size[1])

        # J_out = J_in * stride
        self.j[0] *= stride[0]
        self.j[1] *= stride[1]

        self.results.append({
            'name': layer_name,
            'rf': tuple(self.rf),
            'j': tuple(self.j)
        })

    def analyze_model(self, model, input_size=None):
        """Traverses the model and calculates RF for each layer."""
        # Update input size if provided
        if input_size is not None:
            self.input_size = _param_to_tuple(input_size)
            
        self.rf = [1, 1]
        self.j = [1, 1]
        self.results = []
        
        # Special handling for RESMAX models which have multi-band processing
        if hasattr(model, 'make_ip') and hasattr(model, 's1'):
            self._analyze_resmax(model)
        else:
            self._traverse(model)
        return self.results
    
    def _analyze_resmax(self, model):
        """Special analysis for RESMAX models with multi-band processing."""
        # Analyze each stage of RESMAX
        
        # S1 stage (assuming it processes a single scale for RF calculation)
        if hasattr(model, 's1'):
            self._traverse(model.s1, 's1')
        
        # C1 scoring - handle as special case with parallel pooling
        if hasattr(model, 'c1'):
            self._handle_c_scoring(model.c1, 'c1')
            
        # S2 stage  
        if hasattr(model, 's2'):
            self._traverse(model.s2, 's2')
            
        # C2 scoring
        if hasattr(model, 'c2'):
            self._handle_c_scoring(model.c2, 'c2')
            
        # S2b bypass if present
        if hasattr(model, 's2b'):
            # For bypass, we'll calculate RF but note it's a parallel path
            bypass_rf = list(self.rf)
            bypass_j = list(self.j)
            
            # Reset to C1 output state for bypass calculation
            # Find C1 result
            c1_result = next((r for r in self.results if 'c1' in r['name']), None)
            if c1_result:
                bypass_rf = list(c1_result['rf'])
                bypass_j = list(c1_result['j'])
            
            # Temporarily use bypass state
            main_rf, main_j = self.rf, self.j
            self.rf, self.j = bypass_rf, bypass_j
            
            self._traverse(model.s2b, 's2b')
            
            # Handle c2b_score if present
            if hasattr(model, 'c2b_score'):
                self._handle_c_scoring(model.c2b_score, 'c2b_score')
                
            # Handle c2b_seq if present  
            if hasattr(model, 'c2b_seq'):
                self._traverse(model.c2b_seq, 'c2b_seq')
                
            # Restore main path state
            self.rf, self.j = main_rf, main_j
            
        # S3 stage
        if hasattr(model, 's3'):
            self._traverse(model.s3, 's3')
            
        # Global pooling
        if hasattr(model, 'global_pool'):
            if hasattr(model.global_pool, 'pool1'):  # It's another C-scoring layer
                self._handle_c_scoring(model.global_pool, 'global_pool')
            else:
                self._traverse(model.global_pool, 'global_pool')
    
    def _handle_c_scoring(self, module, prefix):
        """Handle C-scoring layers with parallel pooling paths."""
        if not (hasattr(module, 'pool1') and hasattr(module, 'pool2')):
            # Not a C-scoring layer, handle normally
            self._traverse(module, prefix)
            return
            
        # Store the input state
        rf_input = list(self.rf)
        j_input = list(self.j)
        
        # Calculate RF for both pooling paths
        paths = []
        
        # Path 1: pool1
        self.rf = list(rf_input)
        self.j = list(j_input)
        self._update_rf(f"{prefix}.pool1", module.pool1)
        paths.append({'rf': list(self.rf), 'j': list(self.j), 'name': 'pool1'})
        
        # Path 2: pool2  
        self.rf = list(rf_input)
        self.j = list(j_input)
        self._update_rf(f"{prefix}.pool2", module.pool2)
        paths.append({'rf': list(self.rf), 'j': list(self.j), 'name': 'pool2'})
        
        # For HMAX C-scoring, use the MINIMUM RF path since:
        # - The operation takes max across different scale bands
        # - Each pool operates on DIFFERENT scale bands (not sequential)
        # - The effective RF is limited by the most restrictive path
        min_rf_path = min(paths, key=lambda x: x['rf'][0] * x['rf'][1])
        self.rf = min_rf_path['rf']
        self.j = min_rf_path['j']
        
        # Add result for the C-scoring layer
        self.results.append({
            'name': f"{prefix}",
            'rf': tuple(self.rf),
            'j': tuple(self.j)
        })
        
        # Handle other components like resizing layers
        for name, child in module.named_children():
            if name not in ['pool1', 'pool2']:  # Skip pools we already handled
                child_name = f"{prefix}.{name}"
                if len(list(child.children())) > 0:
                    self._traverse(child, child_name)
                else:
                    self._update_rf(child_name, child)

    def _traverse(self, module, prefix=''):
        """Recursive traversal function that understands model structure."""
        
        # Handle ResNet blocks as special cases
        if isinstance(module, (Bottleneck, BasicBlock)):
            # Store the input state
            rf_input = list(self.rf)
            j_input = list(self.j)
            
            # Process main path
            if isinstance(module, Bottleneck):
                # Bottleneck: conv1 -> conv2 -> conv3
                self._update_rf(f"{prefix}.conv1", module.conv1)
                self._update_rf(f"{prefix}.conv2", module.conv2)
                self._update_rf(f"{prefix}.conv3", module.conv3)
            else:  # BasicBlock
                # BasicBlock: conv1 -> conv2
                self._update_rf(f"{prefix}.conv1", module.conv1)
                self._update_rf(f"{prefix}.conv2", module.conv2)
            
            # Store the main path result
            rf_main = list(self.rf)
            j_main = list(self.j)
            
            # Calculate shortcut path
            self.rf = rf_input
            self.j = j_input
            
            if module.downsample is not None:
                # Process downsample layers
                for i, layer in enumerate(module.downsample):
                    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                        self._update_rf(f"{prefix}.downsample.{i}", layer)
            
            # The shortcut path RF
            rf_shortcut = list(self.rf)
            j_shortcut = list(self.j)
            
            # For residual blocks, we need to consider that the addition operation
            # can see information from both paths. The effective RF should account
            # for the maximum reach of either path.
            # However, in practice, the main path dominates the RF calculation
            self.rf[0] = rf_main[0]  # Use main path RF as it's typically larger
            self.rf[1] = rf_main[1]
            self.j = j_main  # Both paths should have same final stride
            
            return # Stop further recursion as we've handled the block
            

        # Generic traversal for other modules
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            
            if len(list(child.children())) > 0:
                # If it's a container, recurse
                self._traverse(child, child_name)
            else:
                # If it's a leaf module (layer), update RF
                self._update_rf(child_name, child)

# --- Analysis Script ---
if __name__ == "__main__":
    models = [
        ('resnet18', resnet18(pretrained=False)),
        ('resnet50', resnet50(pretrained=False)),
        ('alexnet', alexnet(pretrained=False)),
        ('chresmax_v3_2', RESMAX_V2_2(num_classes=1000, ip_scale_bands=3, classifier_input_size=18432, bypass=True)),
    ]

    analyzer = RFAnalyzer(enable_upper_bound=True)

    for model_name, model in models:
        print(f"\n## Analytical Receptive Field for {model_name}")
        print("| Layer Name         | Receptive Field (H x W) | Cumulative Stride (J) |")
        print("|--------------------|-------------------------|-----------------------|")
        
        analysis_results = analyzer.analyze_model(model, input_size=322)
        
        # Filter out intermediate pooling results to show only key stages
        key_results = []
        for result in analysis_results:
            # print(result)  # Debug print to see all results
            key_results.append(result)

        for result in key_results:
            rf_str = f"{result['rf'][0]} x {result['rf'][1]}"
            j_str = f"{result['j'][0]} x {result['j'][1]}"
            print(f"| {result['name']:<25} | {rf_str:<23} | {j_str:<21} |")
            