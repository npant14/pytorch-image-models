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

# Default configuration constants
DEFAULT_CURV_SETS = [1, 2]
DEFAULT_ROTATIONS = list(range(1, 8))  # 1 to 7
DEFAULT_SCALES = [0.4, 0.6, 0.8, 1.0]
DEFAULT_IMG_SIZE = 322 # TODO: receptive field size 
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
        
        print("setup Pasupathy experiment -- ready to run")
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
    
    def layer_feature_nested(self, features, layer):
        tensor_feature = features[layer]
        
        if type(tensor_feature) is tuple or type(tensor_feature) is list:
            tensor_feature = tensor_feature[0]
            # If it's still nested, get the first element again
            if type(tensor_feature) is tuple or type(tensor_feature) is list:
                tensor_feature = tensor_feature[0]
                
        return tensor_feature

    def extract_center_activations(self, layer_features, img, layer, neuron_idx=None):
        """
        Helper function to extract center activations from a layer for a given image.
        
        Args:
            layer_features: FeatureExtractor instance
            img: Input image tensor
            layer: Layer name
            neuron_idx: Specific neuron index (if None, returns all neurons)
            
        Returns:
            Center activations (single value if neuron_idx specified, tensor if all neurons)
        """
        with torch.no_grad():  # Disable gradients to save memory
            features = layer_features(torch.unsqueeze(img, 0).to(self.device))
            tensor_feature = self.layer_feature_nested(features, layer)
            
            # Handle different tensor shapes for different layer types
            if len(tensor_feature.shape) == 4:
                # Convolutional layers: (batch_size, channels, height, width)
                batch_size, channels, height, width = tensor_feature.shape
                center_h, center_w = height // 2, width // 2
                
                if neuron_idx is not None:
                    # Return single neuron activation
                    center_activation = tensor_feature[0, neuron_idx, center_h, center_w].item()
                else:
                    # Return all neurons' activations
                    center_activation = tensor_feature[0, :, center_h, center_w].detach().cpu()
            else:
                # Skip layers with unsupported tensor shapes (e.g., 1D, 3D, 5D+)
                print(f"Warning: Skipping layer '{layer}' with unsupported tensor shape: {tensor_feature.shape}")
                return None
            
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
            if len(scales) > 1:
                slope, intercept = np.polyfit(scales, avg_activities, 1)
                r_squared = np.corrcoef(scales, avg_activities)[0, 1] ** 2
            else:
                slope, intercept, r_squared = 0, 0, 0
            
            neuron_scale_analysis[neuron_idx] = {
                'preferred_rotation': preferred_rot,
                'scales': scales,
                'activities': avg_activities,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'max_activity_at_preferred': pref_data['max_activity']
            }
        
        return neuron_scale_analysis

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
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
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
        
        plt.show()
        
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

        # TODO: abs value? check paper
        
        # TODO: only a small subset of neurons showed systematic shifts in their stimulus preferences, as indicated by significant linear regression slopes (N = 13/80; black).
        
        # TODO: Percentage across layers
        
        # TODO: trend of percentage of neurons that are scale invariant across layers
        
        # Step 0: Select neuron: filter activities, calculate orientation
        print("\n" + "="*60)
        print("STEP 0: Selecting responsive neurons")
        print("="*60)
        included_neurons, rejected_neurons, rejection_reasons = self.select_neurons(self.layer)
        
        if not included_neurons:
            print("No neurons passed the selection criteria. Aborting analysis.")
            return {}

        # Step 1: Find preferred orientations for each neuron
        print("\n" + "="*60)
        print("STEP 1: Finding preferred orientations for each neuron")
        print("="*60)
        
        # Load sample images with cropping already applied
        sample_imgs = load_images(self.data_dir, self.curv_sets[0], self.rotations[0], self.img_size, self.rf_size)
        
        if not sample_imgs:
            raise ValueError("No images found. Check data directory and parameters.")
        
        preferred_orientations = self.find_preferred_orientations(
            self.model, sample_imgs, self.layer, canonical_scale=1.0
        )
        
        # Filter preferred_orientations to only include selected neurons
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
        
        histogram_path = os.path.join(self.outdir, f'{self.layer.replace(".", "_")}.png')
        summary_stats = self.create_score_histograms(neuron_scale_analysis, histogram_path)
        
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

    def save_comprehensive_results(self, comprehensive_results, save_path=None):
        """
        Save comprehensive results to files for later analysis.
        
        Args:
            comprehensive_results: Results from run_neuron_analysis()
            save_path: Base path for saving files (default: outdir)
        """
        if save_path is None:
            save_path = self.outdir
        
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
        
        print(f"DEBUG: About to create CSV file at: {csv_path}")
        print(f"DEBUG: Individual data length: {len(individual_data)}")
        
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['neuron_index', 'preferred_rotation', 'scale_invariance_score', 
                               'r_squared', 'max_activity'])
                
                rows_written = 0
                for neuron_data in individual_data:
                    try:
                        writer.writerow([
                            neuron_data['neuron_index'],
                            neuron_data['preferred_rotation'],
                            neuron_data['scale_invariance_score'],
                            neuron_data['r_squared'],
                            neuron_data['max_activity']
                        ])
                        rows_written += 1
                    except KeyError as e:
                        print(f"DEBUG: Missing key in neuron_data: {e}")
                        print(f"DEBUG: Available keys: {list(neuron_data.keys())}")
                        raise
                    except Exception as e:
                        print(f"DEBUG: Error writing row for neuron {neuron_data.get('neuron_index', 'unknown')}: {e}")
                        raise
                
                print(f"DEBUG: Successfully wrote {rows_written} rows to CSV")
                
        except Exception as e:
            print(f"ERROR: Failed to create CSV file: {e}")
            import traceback
            traceback.print_exc()
            raise
        
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

    def pasupathy(self, model, images, scale, layer):
        # Store all neuron responses for each image
        # activations will be a list of tensors, each with shape (channels,)
        activations = []
        print(f"Processing {len(images)} images at scale {scale}")
        
        # Create FeatureExtractor once instead of for each image
        layer_features = FeatureExtractor(model, [layer])
        
        for i, img in enumerate(images):
            if i % self.progress_interval == 0:  # Progress indicator
                print(f"Processing image {i+1}/{len(images)}")
                
            size = int(scale * self.img_size)  # Scale relative to img_size, not original img size
            img = self.resize_image(img, size)
            
            # Extract center activations for all neurons
            center_activations = self.extract_center_activations(layer_features, img, layer)
            
            if center_activations is None:
                return None, None
                
            # Store the full neuron population response
            activations.append(center_activations)

        # Stack all activations to get shape (num_images, num_channels)
        activations_tensor = torch.stack(activations)  # Shape: (num_images, num_channels)
        
        # Compute median response for each neuron across all presentations
        # activations_tensor shape: (num_images, num_channels)
        # median across dim=0 (images) gives us (num_channels,) - one value per neuron
        neuron_medians = torch.median(activations_tensor, dim=0)[0]  # Shape: (num_channels,)
        
        # Find the neuron with maximum median response
        best_neuron_idx = torch.argmax(neuron_medians).item()
        
        # Return both the best neuron index and the full vector for future analysis
        return best_neuron_idx, neuron_medians

    def run(self):
        slopes = np.zeros((0))
        total_combinations = len(self.curv_sets) * len(self.rotations)
        current_combination = 0
        
        # Store all neuron population data across all combinations
        all_neuron_populations = []
        
        for curv_set in self.curv_sets:
            for rot in self.rotations:
                current_combination += 1
                print(f"\n{'='*50}")
                print(f"Processing combination {current_combination}/{total_combinations}: curv={curv_set}, rot={rot}")
                print(f"{'='*50}")
                
                # Load images with RF cropping applied
                imgs = load_images(self.data_dir, curv_set, rot, self.img_size, self.rf_size)
                print(f"Loaded {len(imgs)} images")
                
                selections = []
                neuron_populations = []  # Store the full neuron response vectors for this combination
                for scale in self.scales:
                    print(f"\n--- Processing scale {scale} ---")
                    selection, neuron_medians = self.pasupathy(self.model, imgs, scale, self.layer)
                    
                    # Handle the case where the layer was skipped due to unsupported shape
                    if selection is None:
                        print(f"Layer {self.layer} was skipped due to unsupported tensor shape")
                        return None, None
                    
                    selections.append(selection)
                    neuron_populations.append(neuron_medians)
                    print(f"Selection for scale {scale}: {selection}")
                
                # Store this combination's neuron populations
                all_neuron_populations.append({
                    'curv_set': curv_set,
                    'rotation': rot,
                    'selections': selections,
                    'neuron_populations': neuron_populations
                })
                
                # Fit line using the configured scales
                slope, _ = np.polyfit(self.scales, selections, 1)
                slopes = np.append(slopes, slope)
                print(f"Slope for this combination: {slope}")
                
                # Clean up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        final_mean = np.mean(slopes)
        print(f"\n{'='*50}")
        print(f"FINAL RESULT: Mean slope = {final_mean}")
        print(f"{'='*50}")
        
        # Return both the final mean slope and all the neuron populations for future analysis
        # all_neuron_populations contains data for each combination (curv_set, rotation)
        # Each entry contains: curv_set, rotation, selections, neuron_populations
        # neuron_populations is a list of 4 tensors (one for each scale: 0.4, 0.6, 0.8, 1)
        # Each tensor has shape (num_channels,) representing median response per neuron
        
        # Also create a summary of the neuron population data for easier access
        neuron_summary = {
            'final_mean_slope': final_mean,
            'total_combinations': len(all_neuron_populations),
            'scales': self.scales,
            'curv_sets': self.curv_sets,
            'rotations': self.rotations,
            'combinations': all_neuron_populations
        }
        
        return final_mean, neuron_summary

    def get_baseline_activity(self, layer, num_repeats=5, shift_pixels=2, noise_std=0.05):
        """
        Calculate the baseline activity of neurons using low-contrast random noise stimuli with small shifts.
        """
        print("Calculating baseline activity with low-contrast random noise...")
        layer_features = FeatureExtractor(self.model, [layer])
        baseline_activities = []
        for _ in range(num_repeats):
            try:
                # Generate low-contrast random noise image at full image size
                # very close to black, and low contrast + little bit noise
                # TODO: get mean response for different std in response to stimuli
                noise = torch.clamp(torch.normal(mean=0.5, std=noise_std, size=(3, self.img_size, self.img_size)), 0, 1)
                
                # Apply RF cropping if needed (same as other images)
                if self.rf_size < self.img_size:
                    crop_transform = CenterCropPad(output_size=(self.img_size, self.img_size), crop_size=self.rf_size, mode='constant')
                    noise = crop_transform(noise)
                
                # Apply a random shift
                shift_x = np.random.randint(-shift_pixels, shift_pixels + 1)
                shift_y = np.random.randint(-shift_pixels, shift_pixels + 1)
                shifted_image = transforms.functional.affine(noise, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0)
                
                # Extract center activations for all neurons
                center_activations = self.extract_center_activations(layer_features, shifted_image, layer)
                
                if center_activations is None:
                    raise ValueError(f"Unsupported tensor shape for layer {layer}")
                
                baseline_activities.append(center_activations)
                
            except Exception as e:
                print(f"Error calculating baseline activity for layer {layer}: {e}")
                raise e

        if not baseline_activities:
            return None
        # Average across repetitions
        mean_baseline_activity = torch.mean(torch.stack(baseline_activities), dim=0)
        return mean_baseline_activity

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
                # Load images with RF cropping applied
                imgs = load_images(self.data_dir, curv_set, rot, self.img_size, self.rf_size)
                total_images += len(imgs)
                for img in imgs:
                    try:
                        # Use full image size for resize (cropping already applied in load_images)
                        img_resized = self.resize_image(img, self.img_size)
                        
                        # Extract center activations for all neurons
                        center_activations = self.extract_center_activations(layer_features, img_resized, layer)
                        
                        all_responses.append(center_activations)
                        
                    except Exception as e:
                        print(f"Error processing image for layer {layer}: {e}")
                        raise e

        if not all_responses:
            return None, 0
            
        return torch.stack(all_responses), total_images

    def select_neurons(self, layer, min_repeats=5, selectivity_threshold=4):
        """
        Selects neurons based on their response properties.
        """
        print("Starting neuron selection...")
                
        # Step 1: Get all shape responses and check trial count
        all_responses, num_trials = self.get_all_shape_responses(layer)
        if all_responses is None:
            print("Could not get shape responses. Aborting neuron selection.")
            return None, None, None

        if num_trials < min_repeats:
            print(f"Insufficient trials ({num_trials}) for all neurons. Minimum required: {min_repeats}.")
            # Depending on desired behavior, you might want to stop or just warn.
            # For now, we'll just create empty lists and return.
            return [], list(range(all_responses.shape[1])), {"insufficient repeats": list(range(all_responses.shape[1]))}

        # Step 2: Calculate baseline firing
        baseline_rate_per_neuron = self.get_baseline_activity(layer)
        if baseline_rate_per_neuron is None:
            print("Could not calculate baseline activity. Aborting neuron selection.")
            raise ValueError("Failed to calculate baseline activity")

        # Step 3: Find maximum evoked response for each neuron
        max_response_per_neuron = torch.max(all_responses, dim=0)[0]

        # Step 4: Apply selectivity criterion
        included_neurons = []
        rejected_neurons = []
        rejection_reasons = {"insufficient repeats": [], "weak shape selectivity": []}

        num_neurons = all_responses.shape[1]
        for neuron_idx in range(num_neurons):
            max_response = max_response_per_neuron[neuron_idx]
            baseline_rate = baseline_rate_per_neuron[neuron_idx]

            if max_response < selectivity_threshold * baseline_rate:
                rejected_neurons.append(neuron_idx)
                rejection_reasons["weak shape selectivity"].append(neuron_idx)
            else:
                included_neurons.append(neuron_idx)
        
        print(f"Neuron selection complete. Included: {len(included_neurons)}, Rejected: {len(rejected_neurons)}")
        return included_neurons, rejected_neurons, rejection_reasons