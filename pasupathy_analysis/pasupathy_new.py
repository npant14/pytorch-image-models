"""
runs an evaluation based on data from Pasupathy et. al. 2018(?) 
"""
from utils_hmax import FeatureExtractor, CenterCropPad
import torch
torch.manual_seed(1)
import numpy as np
from torchvision import transforms
from PIL import Image
import csv
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from scipy import stats

# Default configuration constants
DEFAULT_CURV_SETS = [1, 2]
DEFAULT_ROTATIONS = list(range(1, 8))  # 1 to 7
DEFAULT_SCALES = [0.4, 0.6, 0.8, 1.0]
DEFAULT_IMG_SIZE = 322
DEFAULT_LAYER = "s3.layer.3.conv3"
DEFAULT_PROGRESS_INTERVAL = 10

def load_images(datadir, curv, rot, size, rf_size=None):
    """
    Load images from directory with optional cropping to receptive field size.
    
    Args:
        datadir: Directory containing images
        curv: Curvature set to load
        rot: Rotation to load
        size: Target image size
        rf_size: Receptive field size for cropping (optional)
    """
    pattern = re.compile(r"subplot_rot=(\d+)_curv=(\d+)_img=(\d+)\.png")
    
    # Define transformation to convert images to tensors
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),  # Converts image to tensor with shape (C, H, W)
    ])
    
    # Set up cropping if needed
    crop_transform = None
    if rf_size is not None and rf_size < size:
        crop_transform = CenterCropPad(output_size=(size, size), crop_size=rf_size, mode='constant')
    
    # List to store matched images as PyTorch tensors
    image_tensors = []

    print(f"Loading images from {datadir} for curv={curv}, rot={rot}")
    if crop_transform is not None:
        print(f"Will crop images from center to {rf_size}x{rf_size} (RF size) and pad back to {size}x{size}")
    
    # Loop through all files in the folder
    for filename in os.listdir(datadir):
        match = pattern.search(filename)
        if match:
            rot_value = int(match.group(1))
            curv_value = int(match.group(2))
            img_value = int(match.group(3))

            if rot_value == rot and curv_value == curv:
                img_path = os.path.join(datadir, filename)
                try:
                    img = Image.open(img_path).convert("RGB")  # Ensure RGB format
                    img_tensor = transform(img)  # Convert to PyTorch tensor
                    
                    # Apply cropping if needed
                    if crop_transform is not None:
                        img_tensor = crop_transform(img_tensor)
                    
                    image_tensors.append((img_value, img_tensor))
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

    # Sort images by img number
    image_tensors.sort()

    # Extract sorted image tensors
    sorted_tensors = [img_tensor for _, img_tensor in image_tensors]
    print(f"Found {len(sorted_tensors)} images")
    return sorted_tensors

class Pasupathy():
    def __init__(self, model, outdir, device, data_dir, rf_size,
                 img_size=DEFAULT_IMG_SIZE,
                 layer=DEFAULT_LAYER, 
                 curv_sets=None,
                 rotations=None,
                 scales=None,
                 progress_interval=DEFAULT_PROGRESS_INTERVAL):
        
        self.model = model
        self.outdir = outdir
        self.device = device
        self.data_dir = data_dir
        self.img_size = img_size
        self.layer = layer
        self.rf_size = rf_size
        
        # Configurable parameters with sensible defaults
        self.curv_sets = curv_sets if curv_sets is not None else DEFAULT_CURV_SETS
        self.rotations = rotations if rotations is not None else DEFAULT_ROTATIONS
        self.scales = scales if scales is not None else DEFAULT_SCALES
        self.progress_interval = progress_interval
        
        self.outdir_figures = os.path.join(self.outdir, 'figures')
        self.outdir_neuron_data = os.path.join(self.outdir, 'neuron_data')
        
        os.makedirs(self.outdir_figures, exist_ok=True)
        os.makedirs(self.outdir_neuron_data, exist_ok=True)
        
        print(f"Configuration: curv_sets={self.curv_sets}, rotations={self.rotations}, scales={self.scales}")

    def resize_image(self, image, size):
        image_size = image.shape[1]
        pil_image = transforms.ToPILImage()(image)
        resize = transforms.Compose([
                            transforms.Resize((size,size)),
                            transforms.Pad(((image_size - size)//2, (image_size - size)//2)),
                            transforms.Resize(self.img_size),
                            transforms.ToTensor()
                        ])
        new_img = resize(pil_image)

        return new_img
    
    def _layer_feature_nested(self, features, layer):
        """
        Handle nested features, but preserve lists of tensors for multi-scale processing.
        
        Returns:
            torch.Tensor or list: Either a single tensor or a list of tensors (for multi-scale layers)
        """
        tensor_feature = features[layer]
        
        # If it's a list/tuple, check if all elements are tensors (multi-scale case)
        if isinstance(tensor_feature, (tuple, list)):
            # Check if all elements are 4D tensors (multi-scale feature maps)
            if all(isinstance(t, torch.Tensor) and len(t.shape) == 4 for t in tensor_feature):
                # Return the list as-is for multi-scale processing
                print(f"Multi-scale layer detected: {len(tensor_feature)} scales")
                return list(tensor_feature)  # Convert tuple to list if needed
            else:
                # Not all tensors or not 4D - unwrap nested structure
                tensor_feature = tensor_feature[0]
                if isinstance(tensor_feature, (tuple, list)):
                    tensor_feature = tensor_feature[0]
        
        return tensor_feature
    
    
    def _extract_center_helper(self, tensor_feature, neuron_idx=None):
        """
        Helper function to extract center activations from a 4D tensor.
        
        Args:
            tensor_feature: 4D tensor with shape (batch, channels, height, width)
            neuron_idx: Specific neuron index (if None, returns all neurons)
            
        Returns:
            Center activation(s) - single value if neuron_idx specified, tensor if all neurons
        """
        batch_size, channels, height, width = tensor_feature.shape
        center_h, center_w = height // 2, width // 2
        
        if neuron_idx is not None:
            # Return single neuron activation
            return tensor_feature[0, neuron_idx, center_h, center_w].item()
        else:
            # Return all neurons' activations
            return tensor_feature[0, :, center_h, center_w].detach().cpu()


    def extract_center_activations(self, layer_features, img, layer, neuron_idx=None):
        """
        Helper function to extract center activations from a layer for a given image.
        Handles both single-scale and multi-scale layers.
        
        Args:
            layer_features: FeatureExtractor instance
            img: Input image tensor
            layer: Layer name
            neuron_idx: Specific neuron index (if None, returns all neurons)
                For multi-scale layers, this indexes into the concatenated neuron array
                
        Returns:
            Center activations (single value if neuron_idx specified, tensor if all neurons)
        """
        with torch.no_grad():  # Disable gradients to save memory
            features = layer_features(torch.unsqueeze(img, 0).to(self.device))
            tensor_feature = self._layer_feature_nested(features, layer)
            
            # Check if we have a list of tensors (multi-scale layer)
            if isinstance(tensor_feature, list):
                print(f"Processing multi-scale layer with {len(tensor_feature)} scales")
                
                all_center_activations = []
                
                for scale_idx, scale_tensor in enumerate(tensor_feature):
                    if len(scale_tensor.shape) == 4:
                        # Use helper to extract center from this scale
                        center_acts = self._extract_center_helper(scale_tensor, neuron_idx=None)
                        all_center_activations.append(center_acts)
                
                # Concatenate all scales along channel dimension
                if all_center_activations:
                    # Stack and concatenate: e.g., 4 scales × 96 channels = 384 total neurons
                    center_activation = torch.cat(all_center_activations, dim=0)  # Shape: (total_channels,)
                    
                    # If specific neuron requested, index into concatenated tensor
                    if neuron_idx is not None:
                        center_activation = center_activation[neuron_idx].item()
                    
            # Single tensor case (standard convolutional layer)
            elif len(tensor_feature.shape) == 4:
                center_activation = self._extract_center_helper(tensor_feature, neuron_idx)
            
            # Transformer case (e.g., Vision Transformer)
            elif len(tensor_feature.shape) == 3:
                # Shape: (batch, num_patches, embed_dim)
                # For ViT, we can use the class token (first token) or average over spatial tokens
                batch_size, num_patches, embed_dim = tensor_feature.shape
                
                # Option 1: Use class token (index 0)
                # center_activation = tensor_feature[0, 0, :].detach().cpu()
                
                # Option 2: Use center patch token (more analogous to CNN center)
                # For ViT-B/16 with 224x224 input: 14x14=196 patches + 1 class token = 197 total
                # Skip class token (index 0), reshape remaining to spatial grid
                spatial_tokens = tensor_feature[0, 1:, :]  # Shape: (196, 768)
                grid_size = int((num_patches - 1) ** 0.5)  # 14 for ViT-B/16 @ 224x224
                
                # Reshape to spatial grid: (grid_size, grid_size, embed_dim)
                spatial_grid = spatial_tokens.reshape(grid_size, grid_size, embed_dim)
                
                # Extract center patch
                center_h, center_w = grid_size // 2, grid_size // 2
                center_activation = spatial_grid[center_h, center_w, :].detach().cpu()
                
                if neuron_idx is not None:
                    center_activation = center_activation[neuron_idx].item()
            
            else:
                raise ValueError(f"Unsupported layer feature shape: {tensor_feature.shape}")
            
            # Clear GPU tensors
            del tensor_feature, features
                        
            return center_activation
    
    
    def find_preferred_orientations(self, model, images, layer, canonical_scale=1.0):
        """
        Find the preferred orientation for each neuron (channel) at the center of the feature map.
        
        Args:
            model: The neural network model
            images: List of images for each rotation
            layer: Layer name to analyze
            canonical_scale: Scale to use for finding preferred orientation (default: 1.0)
            
        Returns:
            dict: Dictionary with preferred orientations and activities for each neuron
        """
        print(f"Finding preferred orientations at scale {canonical_scale}")
        
        layer_features = FeatureExtractor(model, [layer])
        neuron_data = {}
        
        # Process each rotation
        for rot_idx, rot in enumerate(self.rotations):
            print(f"Processing rotation {rot} ({rot_idx+1}/{len(self.rotations)})")
            
            # Get images for this rotation (assuming images are organized by rotation)
            # We need to get images for each curvature set and this rotation
            rot_activities = []
            
            for curv_set in self.curv_sets:
                # Load images for this curvature set and rotation with cropping applied
                imgs = load_images(self.data_dir, curv_set, rot, self.img_size, self.rf_size)
                
                for img in imgs:
                    # Resize to canonical scale
                    size = int(canonical_scale * self.img_size)  # Scale relative to img_size, not original img size
                    img_resized = self.resize_image(img, size)
                    
                    # Extract center activations for all neurons
                    center_activations = self.extract_center_activations(layer_features, img_resized, layer)
                    
                    if center_activations is None:
                        return None
                    
                    rot_activities.append(center_activations)

            # Average across all images for this rotation
            if rot_activities:
                avg_activities = torch.mean(torch.stack(rot_activities), dim=0)  # Shape: (channels,)
                
                # Store activities for each neuron
                for neuron_idx in range(avg_activities.shape[0]):
                    if neuron_idx not in neuron_data:
                        neuron_data[neuron_idx] = {'activities': [], 'rotations': []}
                    
                    neuron_data[neuron_idx]['activities'].append(avg_activities[neuron_idx].item())
                    neuron_data[neuron_idx]['rotations'].append(rot)
            else:
                print(f"Warning: No valid images found for rotation {rot}")
        
        # Check if we have any valid neuron data
        if not neuron_data:
            print("Error: No valid neuron data found")
            return None
        
        # Find preferred orientation for each neuron
        preferred_orientations = {}
        for neuron_idx, data in neuron_data.items():
            activities = np.array(data['activities'])
            rotations = np.array(data['rotations'])
            
            # Find rotation with maximum activity
            max_idx = np.argmax(activities)
            preferred_rot = rotations[max_idx]
            max_activity = activities[max_idx]
            
            preferred_orientations[neuron_idx] = {
                'preferred_rotation': preferred_rot,
                'max_activity': max_activity,
                'all_activities': activities,
                'all_rotations': rotations
            }
        
        print(f"Found preferred orientations for {len(preferred_orientations)} neurons")
        return preferred_orientations

    def analyze_neuron_scale_invariance(self, model, images, layer, preferred_orientations):
        """
        Analyze scale invariance for each neuron at its preferred orientation.
        
        Args:
            model: The neural network model
            images: List of images
            layer: Layer name to analyze
            preferred_orientations: Dictionary with preferred orientations for each neuron
            
        Returns:
            dict: Dictionary with scale invariance analysis for each neuron
        """
        print("Analyzing scale invariance for each neuron at preferred orientation")
        
        layer_features = FeatureExtractor(model, [layer])
        neuron_scale_analysis = {}
        
        for neuron_idx, pref_data in preferred_orientations.items():
            preferred_rot = pref_data['preferred_rotation']
            print(f"Analyzing neuron {neuron_idx} at preferred rotation {preferred_rot}")
            
            # Get images for the preferred rotation
            scale_activities = []
            
            for curv_set in self.curv_sets:
                # Load images for this curvature set and preferred rotation with cropping applied
                imgs = load_images(self.data_dir, curv_set, preferred_rot, self.img_size, self.rf_size)
                
                for img in imgs:
                    for scale in self.scales:
                        size = int(scale * self.img_size)  # Scale relative to img_size, not original img size
                        img_resized = self.resize_image(img, size)
                        
                        # Extract center activation for specific neuron
                        center_activation = self.extract_center_activations(layer_features, img_resized, layer, neuron_idx)
                        
                        if center_activation is None:
                            return None
                        
                        scale_activities.append({
                            'scale': scale,
                            'activity': center_activation,
                            'neuron_idx': neuron_idx
                        })
            
            # Group by scale and average
            scale_avg_activities = {}
            for activity_data in scale_activities:
                scale = activity_data['scale']
                activity = activity_data['activity']
                
                if scale not in scale_avg_activities:
                    scale_avg_activities[scale] = []
                scale_avg_activities[scale].append(activity)
            
            # Calculate average activity for each scale
            scales = []
            avg_activities = []
            for scale in sorted(scale_avg_activities.keys()):
                scales.append(scale)
                avg_activities.append(np.mean(scale_avg_activities[scale]))
            
            # Calculate slope (scale invariance score)
            # Only include neurons with sufficient scale data points
            if len(scales) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(scales, avg_activities)
                r_squared = r_value ** 2
                
                neuron_scale_analysis[neuron_idx] = {
                    'preferred_rotation': preferred_rot,
                    'scales': scales,
                    'activities': avg_activities,
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared,
                    'p_value': p_value,
                    'std_err': std_err,
                    'is_significant': p_value < 0.05,
                    'max_activity_at_preferred': pref_data['max_activity']
                }
            else:
                # Skip neurons with insufficient scale data points
                print(f"Skipping neuron {neuron_idx}: only {len(scales)} scale point(s) available")
        
        return neuron_scale_analysis
    

    def plot_slope_distribution_figureA(self, neuron_scale_analysis, save_path=None):
        """
        Creates a stacked histogram showing the distribution of slopes,
        replicating the style of Figure 4A from the paper.
        
        - Zooms in on the x-range: [-1.8, 1.8]
        - Uses a bin width of 0.4, centered at 0
        - Uses rwidth=0.9 for gaps between bars
        """
        
        # --- 1. Define Plot Range and Filter Data ---
        
        # Set the bin edges and plot limits
        # Bins will be [-1.8, -1.4], [-1.4, -1.0], ..., [1.4, 1.8]
        bin_width = 0.4
        plot_range_min = -1.8
        plot_range_max = 1.8
        
        significant_slopes_all = []
        non_significant_slopes_all = []
        
        for neuron_id, data in neuron_scale_analysis.items():
            if data.get('is_significant', False):
                significant_slopes_all.append(data['slope'])
            else:
                non_significant_slopes_all.append(data['slope'])
                
        num_neurons_total = len(significant_slopes_all) + len(non_significant_slopes_all)
        if num_neurons_total == 0:
            print("No neuron data to plot.")
            return

        # Filter the data to the "zoomed" plot range
        significant_slopes_zoomed = [
            s for s in significant_slopes_all if plot_range_min <= s <= plot_range_max
        ]
        non_significant_slopes_zoomed = [
            s for s in non_significant_slopes_all if plot_range_min <= s <= plot_range_max
        ]

        # Get the new count *for normalization*
        num_neurons_zoomed = len(significant_slopes_zoomed) + len(non_significant_slopes_zoomed)
        if num_neurons_zoomed == 0:
            print(f"No neuron data within the plot range [{plot_range_min}, {plot_range_max}] to plot.")
            return
            
        # Calculate weights based on the *zoomed* count
        weights_sig = np.ones_like(significant_slopes_zoomed) / num_neurons_zoomed
        weights_nonsig = np.ones_like(non_significant_slopes_zoomed) / num_neurons_zoomed

        # --- 2. Create the Plot ---

        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Define the bin edges
        bin_edges = np.arange(plot_range_min, plot_range_max + bin_width, bin_width)

        ax.hist(
            [significant_slopes_zoomed, non_significant_slopes_zoomed],
            bins=bin_edges,
            stacked=True,
            weights=[weights_sig, weights_nonsig],
            color=['black', 'lightgray'],
            edgecolor='black',
            rwidth=0.9  # Set bar width to 90% of bin width
        )
        
        # --- 3. Style the Plot (Axes, Labels, Ticks) ---
        
        # Set explicit axis limits and ticks
        ax.set_xlim(plot_range_min, plot_range_max)
        ax.set_ylim(0, 0.6)
        # Set ticks as requested
        ax.set_xticks(np.arange(-1.6, 1.6 + 0.8, 0.8)) # Ticks at -1.6, -0.8, 0.0, 0.8, 1.6
        ax.set_yticks([0.00, 0.20, 0.40, 0.60])
        ax.set_yticklabels(['0.00', '0.20', '0.40', '0.60']) # Ensure formatting

        # Calculate median on ALL data
        all_slopes = significant_slopes_all + non_significant_slopes_all
        median_slope = np.median(all_slopes)
        
        # Only plot the median arrow if it's within our zoomed range
        if plot_range_min <= median_slope <= plot_range_max:
            ax.plot(median_slope, 0.55, 'v', color='gray', markersize=12, clip_on=False, zorder=4)
        
        # Update title to show N of plotted neurons and total N
        ax.set_title(
            f'A scale test\nN={num_neurons_zoomed} (of {num_neurons_total} total)', 
            loc='left', 
            fontsize=16, 
            weight='bold'
        )
        ax.set_xlabel('slope [Δ tuning centroid]', fontsize=14)
        ax.set_ylabel('proportion of neurons\n(within plotted range)', fontsize=14)

        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a text note about excluded data
        num_excluded = num_neurons_total - num_neurons_zoomed
        if num_excluded > 0:
            ax.text(
                1.0, 1.02, 
                f'*Excluded {num_excluded} neurons outside range [{plot_range_min}, {plot_range_max}]', 
                transform=ax.transAxes, 
                ha='right', 
                fontsize=9, 
                style='italic'
            )
        
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout for text

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        # plt.show() # Uncomment to display the plot
        plt.close()

    def create_score_histograms(self, neuron_scale_analysis, save_path=None):
        """
        Create histograms showing the distribution of scale invariance scores.
        
        Args:
            neuron_scale_analysis: Dictionary with scale invariance analysis for each neuron
            save_path: Path to save the histogram plots
            
        Returns:
            dict: Summary statistics of the score distributions
        """
        print("Creating histograms for score distributions")
        
        # Extract scores and other metrics
        slopes = []
        r_squared_values = []
        max_activities = []
        preferred_rotations = []
        
        for neuron_idx, data in neuron_scale_analysis.items():
            slopes.append(data['slope'])
            r_squared_values.append(data['r_squared'])
            max_activities.append(data['max_activity_at_preferred'])
            preferred_rotations.append(data['preferred_rotation'])
        
        # Create figure with subplots
        _, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Slope distribution
        axes[0, 0].hist(slopes, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        axes[0, 0].axvline(np.mean(slopes), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(slopes):.3f}')
        axes[0, 0].axvline(np.median(slopes), color='green', linestyle='--', 
                          label=f'Median: {np.median(slopes):.3f}')
        axes[0, 0].set_xlabel('Scale Invariance Score (Slope)')
        axes[0, 0].set_ylabel('Proportion of Neurons')
        axes[0, 0].set_title('Distribution of Scale Invariance Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. R-squared distribution
        axes[0, 1].hist(r_squared_values, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(np.mean(r_squared_values), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(r_squared_values):.3f}')
        axes[0, 1].set_xlabel('R-squared (Goodness of Fit)')
        axes[0, 1].set_ylabel('Number of Neurons')
        axes[0, 1].set_title('Distribution of R-squared Values')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Max activity distribution
        axes[1, 0].hist(max_activities, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].axvline(np.mean(max_activities), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(max_activities):.3f}')
        axes[1, 0].set_xlabel('Maximum Activity at Preferred Orientation')
        axes[1, 0].set_ylabel('Number of Neurons')
        axes[1, 0].set_title('Distribution of Maximum Activities')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Preferred rotation distribution
        rotation_counts = {}
        for rot in preferred_rotations:
            rotation_counts[rot] = rotation_counts.get(rot, 0) + 1
        
        rotations = list(rotation_counts.keys())
        counts = list(rotation_counts.values())
        
        axes[1, 1].bar(rotations, counts, alpha=0.7, color='gold', edgecolor='black')
        axes[1, 1].set_xlabel('Preferred Rotation')
        axes[1, 1].set_ylabel('Number of Neurons')
        axes[1, 1].set_title('Distribution of Preferred Rotations')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Neuron Population Analysis - Scale Invariance', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Histograms saved to {save_path}")
        plt.close()  # Close figure to prevent memory warnings
                
        # Calculate summary statistics
        summary_stats = {
            'total_neurons': len(slopes),
            'slope_mean': np.mean(slopes),
            'slope_std': np.std(slopes),
            'slope_median': np.median(slopes),
            'r_squared_mean': np.mean(r_squared_values),
            'r_squared_std': np.std(r_squared_values),
            'max_activity_mean': np.mean(max_activities),
            'max_activity_std': np.std(max_activities),
            'preferred_rotations': rotation_counts
        }
        
        return summary_stats

    def run_neuron_analysis(self):
        """
        Run the complete neuron-by-neuron analysis with preferred orientation detection.
        
        Returns:
            dict: Complete analysis results including individual neuron data and summary statistics
        """
        print("Starting comprehensive neuron analysis...")
        print(f"Configuration: curv_sets={self.curv_sets}, rotations={self.rotations}, scales={self.scales}")
        
        # TODO: Percentage across layers
        
        # TODO: trend of percentage of neurons that are scale invariant across layers
        
        # Step 0: Select neuron: filter activities, calculate orientation
        print("\n" + "="*60)
        print("STEP 0: Selecting responsive neurons")
        print("="*60)
        included_neurons, rejected_neurons, rejection_reasons = self.select_neurons(self.layer)
        
        # Step 1: Find preferred orientations for each neuron
        print("\n" + "="*60)
        print("STEP 1: Finding preferred orientations for each neuron")
        print("="*60)
        
        sample_imgs = load_images(self.data_dir, self.curv_sets[0], self.rotations[0], self.img_size, self.rf_size)
        
        preferred_orientations = self.find_preferred_orientations(
            self.model, sample_imgs, self.layer, canonical_scale=1.0
        )
        
        preferred_orientations = {k: v for k, v in preferred_orientations.items() if k in included_neurons}

        # Step 2: Analyze scale invariance for each neuron at its preferred orientation
        print("\n" + "="*60)
        print("STEP 2: Analyzing scale invariance at preferred orientations")
        print("="*60)
        
        neuron_scale_analysis = self.analyze_neuron_scale_invariance(
            self.model, sample_imgs, self.layer, preferred_orientations
        )
        
        # Step 3: Create histograms and summary statistics
        print("\n" + "="*60)
        print("STEP 3: Creating histograms and summary statistics")
        print("="*60)
        
        histogram_path = os.path.join(self.outdir_figures, f'{self.layer.replace(".", "_")}.png')
        summary_stats = self.create_score_histograms(neuron_scale_analysis, histogram_path)
        self.plot_slope_distribution_figureA(neuron_scale_analysis,
            save_path=os.path.join(self.outdir_figures, f'{self.layer.replace(".", "_")}_figureA.png'))
        
        # Step 4: Compile comprehensive results
        print("\n" + "="*60)
        print("STEP 4: Compiling comprehensive results")
        print("="*60)

        comprehensive_results = {
            'experiment_config': {
                'curv_sets': self.curv_sets,
                'rotations': self.rotations,
                'scales': self.scales,
                'layer': self.layer,
                'img_size': self.img_size
            },
            'selected_neurons': included_neurons,
            'preferred_orientations': preferred_orientations,
            'neuron_scale_analysis': neuron_scale_analysis,
            'summary_statistics': summary_stats,
            'individual_neuron_data': []
        }
        
        # Create detailed individual neuron data for easy analysis
        for neuron_idx, data in neuron_scale_analysis.items():
            neuron_data = {
                'neuron_index': neuron_idx,
                'preferred_rotation': data['preferred_rotation'],
                'scale_invariance_score': data['slope'],
                'r_squared': data['r_squared'],
                'p_value': data['p_value'],
                'std_err': data['std_err'],
                'is_significant': data['is_significant'],
                'max_activity': data['max_activity_at_preferred'],
                'scale_activity_curve': {
                    'scales': data['scales'],
                    'activities': data['activities']
                }
            }
            comprehensive_results['individual_neuron_data'].append(neuron_data)
        
        # Print summary
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE - SUMMARY")
        print(f"{'='*60}")
        print(f"Total neurons analyzed: {summary_stats['total_neurons']} (selected from a larger pool)")
        print(f"Neuron selection - Included: {len(included_neurons)}, Rejected: {len(rejected_neurons)}, Reasons: {rejection_reasons}")
        print(f"Mean scale invariance score: {summary_stats['slope_mean']:.4f} ± {summary_stats['slope_std']:.4f}")
        print(f"Median scale invariance score: {summary_stats['slope_median']:.4f}")
        print(f"Mean R-squared: {summary_stats['r_squared_mean']:.4f} ± {summary_stats['r_squared_std']:.4f}")
        print(f"Mean max activity: {summary_stats['max_activity_mean']:.4f} ± {summary_stats['max_activity_std']:.4f}")
        print(f"Histograms saved to: {histogram_path}")
        print(f"{'='*60}")
        
        return comprehensive_results

    def save_comprehensive_results(self, comprehensive_results):
        """
        Save comprehensive results to files for later analysis.
        
        Args:
            comprehensive_results: Results from run_neuron_analysis()
            save_path: Base path for saving files (default: outdir)
        """
        
        save_path = self.outdir_neuron_data
        
        # Save as JSON (human readable)
        json_path = os.path.join(save_path, f'{self.layer.replace(".", "_")}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_json_serializable(obj):
            """Recursively convert numpy types to JSON-serializable types."""
            if isinstance(obj, dict):
                return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            else:
                # Fallback: try to convert to string
                return str(obj)
        
        json_results = convert_to_json_serializable(comprehensive_results)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save as pickle (preserves exact data types)
        pickle_path = os.path.join(save_path, f'{self.layer.replace(".", "_")}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(comprehensive_results, f)
        
        # Save summary statistics as CSV
        csv_path = os.path.join(save_path, f'{self.layer.replace(".", "_")}.csv')
        individual_data = comprehensive_results['individual_neuron_data']
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['neuron_index', 'preferred_rotation', 'scale_invariance_score', 
                        'r_squared', 'p_value', 'std_err', 'is_significant', 'max_activity'])
            
            for neuron_data in individual_data:
                writer.writerow([
                    neuron_data['neuron_index'],
                    neuron_data['preferred_rotation'],
                    neuron_data['scale_invariance_score'],
                    neuron_data['r_squared'],
                    neuron_data['p_value'],
                    neuron_data['std_err'],
                    neuron_data['is_significant'],
                    neuron_data['max_activity']
                ])
            
        
        print(f"Results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Pickle: {pickle_path}")
        print(f"  CSV: {csv_path}")
        
        # Verify CSV file was actually created
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path)
            print(f"DEBUG: CSV file exists, size: {file_size} bytes")
        else:
            print(f"ERROR: CSV file was not created at {csv_path}")

    def get_all_shape_responses(self, layer):
        """
        Get responses for all shape stimuli across all conditions.
        """
        print("Getting all shape responses...")
                        
        layer_features = FeatureExtractor(self.model, [layer])
        all_responses = []

        total_images = 0
        for curv_set in self.curv_sets:
            for rot in self.rotations:
                imgs = load_images(self.data_dir, curv_set, rot, self.img_size, self.rf_size)
                total_images += len(imgs)
                for img in imgs:
                    img_resized = self.resize_image(img, self.img_size)
                    center_activations = self.extract_center_activations(layer_features, img_resized, layer)
                    
                    all_responses.append(center_activations)

        return torch.stack(all_responses), total_images

    def select_neurons(self, layer, min_repeats=5, z_threshold=2.0):
        """
        Selects neurons based on z-scored responses across stimuli.
        A neuron is considered visually selective if it has at least one stimulus 
        that evokes a response > z_threshold standard deviations above its mean.
        """        
        # Step 1: Get all shape responses
        all_responses, num_trials = self.get_all_shape_responses(layer) # (num_stimuli, num_neurons)
        
        num_neurons = all_responses.shape[1]
        print(f"Analyzing {num_neurons} neurons across {num_trials} stimulus presentations")
        
        # Step 2: Calculate z-scores for each neuron across all stimuli
        # For each neuron, compute: z = (response - mean) / std
        mean_per_neuron = torch.mean(all_responses, dim=0)  # Shape: (num_neurons,)
        std_per_neuron = torch.std(all_responses, dim=0)    # Shape: (num_neurons,)
        
        # Compute z-scores: (num_stimuli, num_neurons)
        z_scores = (all_responses - mean_per_neuron) / (std_per_neuron + 1e-8)
        
        # Step 3: Find maximum z-score for each neuron across all stimuli
        max_z_per_neuron = torch.max(z_scores, dim=0)[0]  # Shape: (num_neurons,)
        
        # Step 4: Select neurons with max z-score >= threshold
        included_neurons = []
        rejected_neurons = []
        rejection_reasons = {
            "low max z-score": [],
        }
        
        for neuron_idx in range(num_neurons):
            max_z = max_z_per_neuron[neuron_idx].item()
            
            # Check for neurons with zero variance (non-responsive)
            if max_z < z_threshold:
                rejected_neurons.append(neuron_idx)
                rejection_reasons["low max z-score"].append(neuron_idx)
            else:
                included_neurons.append(neuron_idx)
        
        # Print statistics
        print(f"\nNeuron Selection Results:")
        print(f"  Included neurons: {len(included_neurons)} ({100*len(included_neurons)/num_neurons:.1f}%)")
        print(f"  Rejected neurons: {len(rejected_neurons)} ({100*len(rejected_neurons)/num_neurons:.1f}%)")
        
        return included_neurons, rejected_neurons, rejection_reasons
