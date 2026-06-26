# Human On Truck

Human On Truck is a post-processing computer vision utility for filtering detected humans and keeping only the humans who are likely standing on top of a truck.

The project is designed to work after an object detection / segmentation model has already produced:

- Truck segmentation masks
- Human bounding boxes
- Optional class IDs for different human-related classes

The core logic checks whether the lower body or bottom support region of a human bounding box overlaps with a truck mask. This helps reject people standing beside a truck while keeping people actually positioned on top of it.

## Project Purpose

In truck safety monitoring, object detectors may detect humans near trucks, beside trucks, or partially overlapping with trucks. A simple bounding-box overlap is not enough because a person beside the truck can still overlap with the truck mask.

This project uses region-based overlap logic to distinguish:

- Human on top of truck
- Human beside truck
- Human partially touching truck
- Human with insufficient bottom support
- Human detections that should be ignored

## Repository Structure

```text
human-on-truck-main/
│
├── filter_on_top_human (3).py
├── filter_on_top_human.cpp
└── README.md
```

## Files

| File | Description |
|---|---|
| `filter_on_top_human (3).py` | Python implementation of the human-on-truck filtering logic |
| `filter_on_top_human.cpp` | C++ implementation converted from the Python logic |
| `README.md` | Project documentation |

## Main Idea

The algorithm does not run object detection by itself.

Instead, it receives detection results from another model, such as YOLO segmentation, and filters the human detections.

Input:

```text
Frame image
Truck masks
Human bounding boxes
```

Output:

```text
Only human boxes classified as on top of a truck
```

## Algorithm Overview

For each human bounding box:

1. Validate the human box.
2. Optionally adjust the logic box depending on class ID.
3. Compare the human box with each truck mask.
4. Compute overlap in multiple regions:
   - Full human box
   - Lower body region
   - Bottom support band
   - Left side strip
   - Right side strip
5. Select the best matching truck.
6. Accept the human only if the overlap pattern looks like the human is supported by the truck.
7. Reject humans whose overlap is mostly side contact or too weak.

## Why Region-Based Filtering?

A full bounding-box overlap can be misleading.

For example:

```text
Person beside truck:
- Full box may overlap with the truck mask
- Side overlap may be high
- Bottom support may be weak

Person on top of truck:
- Bottom band overlaps with the truck
- Lower body overlaps with the truck
- Bottom support dominates side contact
```

So the algorithm focuses heavily on bottom support and lower-body overlap.

## Supported Modes

The function supports two modes:

| Mode | Description |
|---|---|
| `relaxed` | More tolerant. Useful when detections or masks are noisy |
| `restricted` | Stricter. Useful when false positives must be reduced |

### Relaxed Mode

Relaxed mode accepts weaker overlap and does not erode the truck mask.

Main settings:

```text
MIN_ASSOC_OVERLAP_PERCENT = 2.0
FULL_OVERLAP_PERCENT_STRICT = 6.0
LOWER_OVERLAP_PERCENT_RELAXED = 3.0
BOTTOM_SUPPORT_PERCENT_MIN = 2.5
BOTTOM_DOMINANCE_RATIO = 1.20
SIDE_OVERLAP_PERCENT_MAX_STRICT = 45.0
ERODE_TRUCK_MASK = False
```

### Restricted Mode

Restricted mode is more conservative and erodes the truck mask before evaluation.

Main settings:

```text
MIN_ASSOC_OVERLAP_PERCENT = 4.0
FULL_OVERLAP_PERCENT_STRICT = 24.0
LOWER_OVERLAP_PERCENT_RELAXED = 18.0
BOTTOM_SUPPORT_PERCENT_MIN = 18.0
BOTTOM_DOMINANCE_RATIO = 5.0
SIDE_OVERLAP_PERCENT_MAX_STRICT = 6.0
ERODE_TRUCK_MASK = True
```

## Python Usage

The Python function is defined in:

```text
filter_on_top_human (3).py
```

Function:

```python
filter_on_top_humans(frame, truck_masks, human_bboxes, mode="relaxed")
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `frame` | `np.ndarray` | Input image frame |
| `truck_masks` | `list` | List of truck masks or dictionaries containing truck masks |
| `human_bboxes` | `list` | List of human bounding boxes or detection dictionaries |
| `mode` | `str` | Either `"relaxed"` or `"restricted"` |

### Return Value

The function returns a list containing only the original human detections classified as on top of a truck.

```python
filtered_humans = filter_on_top_humans(
    frame,
    truck_masks,
    human_bboxes,
    mode="relaxed"
)
```

## Accepted Python Input Formats

### Truck Masks

Truck masks can be passed directly:

```python
truck_masks = [
    truck_mask_1,
    truck_mask_2
]
```

Each truck mask should be a 2D array with the same height and width as the frame.

You can also pass dictionaries:

```python
truck_masks = [
    {
        "mask": truck_mask_1,
        "conf": 0.91
    },
    {
        "mask": truck_mask_2,
        "conf": 0.87
    }
]
```

### Human Bounding Boxes

Human boxes can be simple lists:

```python
human_bboxes = [
    [100, 50, 180, 250],
    [300, 80, 390, 260]
]
```

Or dictionaries:

```python
human_bboxes = [
    {
        "bbox": [100, 50, 180, 250],
        "class_id": 1
    },
    {
        "xyxy": [300, 80, 390, 260],
        "class_id": 2
    }
]
```

The function supports these box keys:

```text
raw_box
xyxy
bbox
```

The function supports these class keys:

```text
class_id
cls
```

## Python Example

```python
import cv2
import numpy as np

from filter_on_top_human import filter_on_top_humans

frame = cv2.imread("frame.jpg")

truck_mask = cv2.imread("truck_mask.png", cv2.IMREAD_GRAYSCALE)

truck_masks = [
    {
        "mask": truck_mask,
        "conf": 0.95
    }
]

human_bboxes = [
    {
        "bbox": [100, 50, 180, 250],
        "class_id": 1
    },
    {
        "bbox": [300, 80, 390, 260],
        "class_id": 2
    }
]

filtered_humans = filter_on_top_humans(
    frame,
    truck_masks,
    human_bboxes,
    mode="relaxed"
)

print("Humans on top of truck:", filtered_humans)
```

## C++ Usage

The C++ implementation is defined in:

```text
filter_on_top_human.cpp
```

The code is wrapped inside the namespace:

```cpp
on_top_human_filter
```

Main function:

```cpp
std::vector<HumanBBox> filterOnTopHumans(
    const cv::Mat& frame,
    const std::vector<TruckMask>& truck_masks,
    const std::vector<HumanBBox>& human_bboxes,
    const std::string& mode = "relaxed"
);
```

## C++ Data Structures

### HumanBBox

```cpp
struct HumanBBox {
    std::vector<double> box;
    int class_id = 1;
};
```

Example:

```cpp
on_top_human_filter::HumanBBox human;
human.box = {100, 50, 180, 250};
human.class_id = 1;
```

### TruckMask

```cpp
struct TruckMask {
    cv::Mat mask;
    double conf = 1.0;
};
```

Example:

```cpp
on_top_human_filter::TruckMask truck;
truck.mask = truck_mask;
truck.conf = 0.95;
```

## C++ Example

```cpp
#include <opencv2/opencv.hpp>
#include "filter_on_top_human.cpp"

int main()
{
    cv::Mat frame = cv::imread("frame.jpg");
    cv::Mat truck_mask = cv::imread("truck_mask.png", cv::IMREAD_GRAYSCALE);

    std::vector<on_top_human_filter::TruckMask> truck_masks = {
        {truck_mask, 1.0}
    };

    std::vector<on_top_human_filter::HumanBBox> humans = {
        {{100, 50, 180, 250}, 1},
        {{300, 80, 390, 260}, 2}
    };

    std::vector<on_top_human_filter::HumanBBox> filtered =
        on_top_human_filter::filterOnTopHumans(
            frame,
            truck_masks,
            humans,
            "relaxed"
        );

    std::cout << "Humans on top of truck: " << filtered.size() << std::endl;

    return 0;
}
```

## Build Instructions

### Linux

```bash
g++ -std=c++17 main.cpp -o human_on_truck `pkg-config --cflags --libs opencv4`
```

Run:

```bash
./human_on_truck
```

### Compile Only the Function File

```bash
g++ -std=c++17 filter_on_top_human.cpp -c `pkg-config --cflags --libs opencv4`
```

### Windows MSVC Example

```cmd
cl /std:c++17 /EHsc main.cpp ^
 /I"C:\opencv\build\include" ^
 /link /LIBPATH:"C:\opencv\build\x64\vc16\lib" opencv_world4xx.lib
```

Make sure the OpenCV DLL path is added to your system `PATH`.

## Dependencies

### Python

```bash
pip install opencv-python numpy
```

### C++

- C++17 or newer
- OpenCV

Required OpenCV modules:

```text
core
imgproc
highgui
imgcodecs
```

## Input Requirements

### Frame

The frame must be a valid image:

```text
Shape: H x W x C
Type: np.ndarray in Python or cv::Mat in C++
```

### Truck Mask

Each truck mask must:

- Be a single-channel 2D mask
- Have the same height and width as the frame
- Contain non-zero pixels where the truck exists

Example:

```text
Frame shape:      1080 x 1920 x 3
Truck mask shape: 1080 x 1920
```

### Human Boxes

Human boxes must use the format:

```text
[x1, y1, x2, y2]
```

Where:

```text
x1, y1 = top-left corner
x2, y2 = bottom-right corner
```

## Class ID Handling

The function uses a special case for:

```text
HOOK_CLASS_ID = 2
```

For class ID `2`, the top part of the box is cropped before overlap evaluation.

This is useful when a class includes extra upper-region content, such as hook-related detections, and the logic should focus more on the lower part of the detected human region.

Related constants:

```text
HOOK_TOP_CROP_FRAC = 0.32
HOOK_MIN_KEEP_FRAC = 0.50
```

## Region Logic

For every human candidate, the algorithm computes these regions:

### Full Box

The entire human bounding box.

Used to check general overlap with the truck mask.

### Lower Body Box

The lower 50 percent of the human box.

```text
LOWER_BODY_FRAC = 0.50
```

Used because a person standing on a truck should have meaningful lower-body overlap with the truck area.

### Bottom Band

A small horizontal band near the bottom of the human box.

```text
BOTTOM_STRIP_HEIGHT = 10
BOTTOM_STRIP_OFFSET_UP = 10
CENTER_WIDTH_FRAC = 0.40
```

The bottom band is used to estimate whether the human is physically supported by the truck.

### Side Strips

Thin left and right side regions of the human box.

```text
SIDE_STRIP_WIDTH_FRAC = 0.18
```

These help detect side-contact cases, where a person is beside the truck rather than standing on it.

## Decision Logic

A human is accepted when:

1. The human has enough association overlap with at least one truck mask.
2. The bottom support is strong enough.
3. The full overlap is strong enough, or lower-body overlap and bottom support are convincing.
4. Side overlap is not dominant.

A human is rejected when:

- The bottom support is too weak
- The side overlap dominates the bottom support
- The overlap is below the selected mode threshold
- The truck mask is invalid
- The human box is malformed
- The human box does not overlap any truck candidate enough

## Candidate Truck Selection

If multiple truck masks are available, the function evaluates each possible truck-human pair.

The selected truck is based on a scoring function that favors:

- Valid accepted result
- Strong bottom contact
- Strong lower-body overlap
- Strong full-body overlap
- Strong association overlap
- Low side overlap

This allows the function to work when multiple trucks are present in the same frame.

## Robustness Features

The function is designed to fail softly.

It safely handles:

- Empty frames
- Empty mask lists
- Empty human lists
- Invalid boxes
- Invalid masks
- Wrong mask shapes
- Non-matching frame and mask sizes
- Empty ROIs
- Bad input types in Python
- Exceptions during per-object processing

Instead of crashing during normal invalid cases, it returns an empty list or skips the bad item.

## Important Constants

```text
BOTTOM_STRIP_HEIGHT = 10
BOTTOM_STRIP_OFFSET_UP = 10
LOWER_BODY_FRAC = 0.50
SIDE_STRIP_WIDTH_FRAC = 0.18
CENTER_WIDTH_FRAC = 0.40
USE_CENTER_BOTTOM_BAND = True

HOOK_CLASS_ID = 2
HOOK_TOP_CROP_FRAC = 0.32
HOOK_MIN_KEEP_FRAC = 0.50
```

## Mode Configuration

### Relaxed

```text
MIN_ASSOC_OVERLAP_PERCENT = 2.0
FULL_OVERLAP_PERCENT_STRICT = 6.0
LOWER_OVERLAP_PERCENT_RELAXED = 3.0
BOTTOM_SUPPORT_PERCENT_MIN = 2.5
BOTTOM_DOMINANCE_RATIO = 1.20
SIDE_OVERLAP_PERCENT_MAX_STRICT = 45.0
LOW_OVERLAP_REJECT_MODE = low_rel_overlap
ERODE_TRUCK_MASK = False
ERODE_KERNEL_SIZE = 3
ERODE_ITERATIONS = 1
```

### Restricted

```text
MIN_ASSOC_OVERLAP_PERCENT = 4.0
FULL_OVERLAP_PERCENT_STRICT = 24.0
LOWER_OVERLAP_PERCENT_RELAXED = 18.0
BOTTOM_SUPPORT_PERCENT_MIN = 18.0
BOTTOM_DOMINANCE_RATIO = 5.0
SIDE_OVERLAP_PERCENT_MAX_STRICT = 6.0
LOW_OVERLAP_REJECT_MODE = low_percent_overlap
ERODE_TRUCK_MASK = True
ERODE_KERNEL_SIZE = 5
ERODE_ITERATIONS = 1
```

## Recommended Workflow

1. Run your detector or segmentation model.
2. Extract truck masks.
3. Extract human bounding boxes.
4. Pass the frame, truck masks, and human boxes into `filter_on_top_humans()` or `filterOnTopHumans()`.
5. Use the returned list as the final human-on-truck detections.

Example pipeline:

```text
Input video frame
        |
        v
Object detection / segmentation model
        |
        v
Truck masks + human boxes
        |
        v
Human-on-truck filtering
        |
        v
Final humans standing on truck
```

## Notes

- This repository does not include model training code.
- This repository does not include a full video inference pipeline.
- The function is intended as a post-processing filter after detection.
- The truck mask and frame must have the same resolution.
- `relaxed` mode is better for noisy detections.
- `restricted` mode is better when false positives are costly.
- The C++ code closely follows the Python implementation.

## Limitations

- The method depends on the quality of the truck mask.
- Poor segmentation masks can cause false positives or false negatives.
- Heavy occlusion can make bottom-support estimation unreliable.
- Very small human boxes may not provide enough region information.
- People standing near the truck edge may be difficult to classify.
- The function only filters existing detections; it does not detect humans or trucks by itself.

## License

No license file is currently included in this repository.
