import tensorflow_datasets as tfds
import numpy as np

ds = tfds.load('imagenet2012_multilabel', split='validation')


num_correct_per_class = {}
num_images_per_class = {}

cnt = 0
for example in ds:
    cnt += 1
    print('  %s\r'%(cnt), end="")
    # We ignore all problematic images
    if example['is_problematic'].numpy():
        continue

    # The label of the image in ImageNet
    cur_class = example['original_label'].numpy()

    # If we haven't processed this class yet, set the counters to 0
    if cur_class not in num_correct_per_class:
        num_correct_per_class[cur_class] = 0
        num_images_per_class[cur_class] = 0

    if cnt % 250 == 0:
        image = example['image']
        ori_label = example['original_label']
        correct_multi_labels = example['correct_multi_labels']
        unclear_multi_labels = example['unclear_multi_labels']
        is_problematic = example['is_problematic']
        print(image.shape)
        print(ori_label)
        print(correct_multi_labels)
        print(unclear_multi_labels)
        print(is_problematic)
        print('--------------------')

    num_images_per_class[cur_class] += 1

    # Get the predictions for this image
    # cur_pred = predictions[example['file_name'].numpy()]
    cur_pred = np.random.randint(1000, size=1)

    # We count a prediction as correct if it is marked as correct or unclear
    # (i.e., we are lenient with the unclear labels)
    if cur_pred in example['correct_multi_labels'].numpy() or cur_pred in example['unclear_multi_labels'].numpy():
        num_correct_per_class[cur_class] += 1
print(" ")



# Check that we have collected accuracy data for each of the 1,000 classes
num_classes = 1000
assert len(num_correct_per_class) == num_classes
assert len(num_images_per_class) == num_classes

# Compute the per-class accuracies and then average them
final_avg = 0
for cid in range(num_classes):
    final_avg += num_correct_per_class[cid] / num_images_per_class[cid]
final_avg /= num_classes

print(final_avg)

