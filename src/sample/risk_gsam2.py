import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert, nms
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from PIL import Image, ImageEnhance

# Add Grounded-SAM-2 path and change working directory
grounded_sam2_path = '/root/ws/src/risk_nav/src/Grounded-SAM-2'
sys.path.append(grounded_sam2_path)
os.chdir(grounded_sam2_path)  # Change working directory to Grounded-SAM-2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

"""
Grounded SAM 2 Configuration for Office Safety Analysis - High Precision Version
"""

# More specific and clear text prompts (separated by category)
OBJECT_PROMPTS = {
    "basic_furniture": "office chair. desk. table. cabinet. shelf. drawer.",
    "electronics": "computer monitor. laptop computer. desktop computer. keyboard. computer mouse. phone. printer. scanner.",
    "lighting": "ceiling light. desk lamp. floor lamp. light fixture. LED panel.",
    "safety_equipment": "fire extinguisher. emergency exit sign. first aid kit. smoke detector.",
    "structure": "ceiling. floor. wall. window. door. column. beam.",
    "hazards": "electrical cable. power cord. extension cord. loose wire. staircase. stairs. step. edge. corner. glass panel. sharp edge.",
    "storage": "cardboard box. storage box. bag. backpack. briefcase. folder. filing cabinet.",
    "consumables": "paper. book. document. pen. pencil. cup. mug. bottle. trash can."
}

# Combine all prompts
ALL_PROMPTS = " ".join(OBJECT_PROMPTS.values())

IMG_PATH = "/root/ws/src/risk_nav/src/sample/office_img.png"

# Model paths (relative paths)
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

# Multi-threshold settings (for improved accuracy)
DETECTION_THRESHOLDS = [
    {"box_threshold": 0.25, "text_threshold": 0.20},  # Low threshold - detect more objects
    {"box_threshold": 0.35, "text_threshold": 0.25},  # Medium threshold - balance
    {"box_threshold": 0.45, "text_threshold": 0.30},  # High threshold - high confidence
]

# Safety categories and color definitions (more detailed) - spatial color classification + risk assessment
SAFETY_CATEGORIES = {
    "floor": {
        "objects": ["floor", "ground", "surface"],
        "color": (139, 69, 19),  # Brown (floor)
        "alpha": 0.5,
        "description": "Floor Area",
        "base_risk": 5  # Base risk level
    },
    "ceiling": {
        "objects": ["ceiling", "roof", "top"],
        "color": (135, 206, 235),  # Sky blue (ceiling)
        "alpha": 0.5,
        "description": "Ceiling Area",
        "base_risk": 2  # Base risk level
    },
    "safe": {
        "objects": ["computer", "monitor", "laptop", "keyboard", "mouse", "chair", "desk", "table", "lamp", "light", "phone", "book", "paper", "pen", "cup", "bottle", "fire extinguisher", "emergency exit", "first aid", "smoke detector", "wall", "window"],
        "color": (0, 255, 0),  # Green (safe)
        "alpha": 0.3,
        "description": "Safe Objects",
        "base_risk": 1  # Base risk level
    },
    "dynamic": {
        "objects": ["door", "corner", "edge", "entrance", "exit", "opening", "passage", "blind spot", "intersection", "junction", "turning", "column", "pillar", "beam"],
        "color": (255, 255, 0),  # Yellow (dynamic environment)
        "alpha": 0.4,
        "description": "Dynamic Environment (Sight-blocked/Entry zones)",
        "base_risk": 15  # Base risk level
    },
    "caution": {
        "objects": ["glass", "panel", "box", "bag", "backpack", "briefcase", "cabinet", "shelf", "drawer", "filing", "step"],
        "color": (255, 165, 0),  # Orange (caution)
        "alpha": 0.4,
        "description": "Caution Objects",
        "base_risk": 10  # Base risk level
    },
    "danger": {
        "objects": ["cable", "wire", "cord", "stairs", "staircase", "sharp", "loose", "electrical", "power", "extension", "obstacle", "hazard"],
        "color": (255, 0, 0),  # Red (danger)
        "alpha": 0.5,
        "description": "Danger Objects",
        "base_risk": 25  # Base risk level
    }
}

# Risk level definitions
RISK_LEVELS = {
    "SAFE": {"range": (0, 10), "color": (0, 255, 0), "description": "Safe"},
    "LOW": {"range": (11, 30), "color": (255, 255, 0), "description": "Low Risk"},
    "MEDIUM": {"range": (31, 60), "color": (255, 165, 0), "description": "Medium Risk"},
    "HIGH": {"range": (61, 85), "color": (255, 100, 0), "description": "High Risk"},
    "CRITICAL": {"range": (86, 100), "color": (255, 0, 0), "description": "Critical Risk"}
}

def calculate_risk_score(category, confidence, box, image_shape):
    """Calculate risk score (0-100 points)"""
    # Base risk (by category)
    base_risk = SAFETY_CATEGORIES[category]["base_risk"]
    
    # Confidence weight (high confidence = more accurate risk)
    confidence_weight = float(confidence) * 20
    
    # Object size weight
    if isinstance(box, torch.Tensor):
        box = box.cpu().numpy()
    
    x1, y1, x2, y2 = box
    object_area = (x2 - x1) * (y2 - y1)
    image_area = image_shape[0] * image_shape[1]
    size_ratio = object_area / image_area
    
    # Increase risk for larger objects (for specific categories)
    if category in ["danger", "caution"]:
        size_weight = size_ratio * 30
    elif category == "dynamic":
        size_weight = size_ratio * 20
    else:
        size_weight = size_ratio * 5
    
    # Apply special rules
    special_multiplier = 1.0
    if category == "danger":
        special_multiplier = 1.5  # 1.5x weight for dangerous objects
    elif category == "dynamic":
        special_multiplier = 1.2  # 1.2x weight for dynamic environments
    
    # Final risk calculation
    risk_score = (base_risk + confidence_weight + size_weight) * special_multiplier
    
    # Normalize to 0-100 range
    risk_score = max(0, min(100, float(risk_score)))
    
    return risk_score

def get_risk_level(risk_score):
    """Return risk level based on risk score"""
    for level, info in RISK_LEVELS.items():
        if info["range"][0] <= risk_score <= info["range"][1]:
            return level
    return "SAFE"

def get_risk_visualization_params(risk_score, risk_level):
    """Return visualization parameters based on risk level"""
    # Color intensity based on risk level
    intensity = min(1.0, float(risk_score) / 100.0)
    
    # Border thickness based on risk level
    border_thickness = int(2 + (float(risk_score) / 100.0) * 6)  # 2-8 range
    
    # Alpha value based on risk level
    alpha = 0.3 + (float(risk_score) / 100.0) * 0.4  # 0.3-0.7 range
    
    return {
        "intensity": float(intensity),
        "border_thickness": border_thickness,
        "alpha": float(alpha),
        "risk_color": RISK_LEVELS[risk_level]["color"]
    }

def preprocess_image(image_path):
    """Improve detection accuracy through image preprocessing"""
    # Load image with PIL
    pil_image = Image.open(image_path)
    
    # Image enhancement
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.2)  # Increase contrast
    
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.1)  # Increase sharpness
    
    # Size adjustment (large images can degrade detection performance)
    max_size = 1024
    if max(pil_image.size) > max_size:
        ratio = max_size / max(pil_image.size)
        new_size = tuple(int(dim * ratio) for dim in pil_image.size)
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.array(pil_image)
    
    return image_array

def multi_threshold_detection(grounding_model, image, device):
    """Detection with multiple thresholds and combination"""
    all_boxes = []
    all_confidences = []
    all_labels = []
    
    print("Performing multi-threshold detection...")
    
    for i, thresholds in enumerate(DETECTION_THRESHOLDS):
        print(f"  Threshold {i+1}: box={thresholds['box_threshold']}, text={thresholds['text_threshold']}")
        
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=ALL_PROMPTS,
            box_threshold=thresholds['box_threshold'],
            text_threshold=thresholds['text_threshold'],
            device=device
        )
        
        if len(boxes) > 0:
            all_boxes.append(boxes)
            all_confidences.append(confidences)
            all_labels.extend(labels)
            print(f"    -> {len(boxes)} objects detected")
        else:
            print(f"    -> No objects detected")
    
    if not all_boxes:
        print("WARNING: No objects detected at any threshold.")
        return torch.empty((0, 4)), torch.empty(0), []
    
    # Combine results
    combined_boxes = torch.cat(all_boxes, dim=0)
    combined_confidences = torch.cat(all_confidences, dim=0)
    
    return combined_boxes, combined_confidences, all_labels

def apply_nms(boxes, confidences, labels, iou_threshold=0.5):
    """Remove duplicates using Non-Maximum Suppression"""
    if len(boxes) == 0:
        return boxes, confidences, labels
    
    print(f"Before NMS: {len(boxes)} objects")
    
    # Apply NMS for each label
    unique_labels = list(set(labels))
    final_boxes = []
    final_confidences = []
    final_labels = []
    
    for label in unique_labels:
        # Find indices for this label
        label_indices = [i for i, l in enumerate(labels) if l == label]
        
        if len(label_indices) == 0:
            continue
            
        label_boxes = boxes[label_indices]
        label_confidences = confidences[label_indices]
        
        # Apply NMS
        if len(label_boxes) > 1:
            keep_indices = nms(label_boxes, label_confidences, iou_threshold)
            keep_indices = keep_indices.cpu().numpy()
            
            final_boxes.extend(label_boxes[keep_indices])
            final_confidences.extend(label_confidences[keep_indices])
            final_labels.extend([label] * len(keep_indices))
        else:
            final_boxes.extend(label_boxes)
            final_confidences.extend(label_confidences)
            final_labels.extend([label])
    
    if final_boxes:
        final_boxes = torch.stack(final_boxes)
        final_confidences = torch.stack(final_confidences)
    else:
        final_boxes = torch.empty((0, 4))
        final_confidences = torch.empty(0)
        final_labels = []
    
    print(f"After NMS: {len(final_boxes)} objects")
    
    return final_boxes, final_confidences, final_labels

def filter_low_confidence(boxes, confidences, labels, min_confidence=0.3):
    """Filter objects with low confidence"""
    if len(boxes) == 0:
        return boxes, confidences, labels
    
    print(f"Before confidence filtering: {len(boxes)} objects")
    
    high_conf_indices = confidences >= min_confidence
    
    filtered_boxes = boxes[high_conf_indices]
    filtered_confidences = confidences[high_conf_indices]
    filtered_labels = [labels[i] for i in range(len(labels)) if high_conf_indices[i]]
    
    print(f"After confidence filtering: {len(filtered_boxes)} objects (threshold: {min_confidence})")
    
    return filtered_boxes, filtered_confidences, filtered_labels

def categorize_object(label):
    """Classify objects into safety categories (including spatial classification)"""
    label_lower = label.lower()
    
    # Floor classification priority
    if any(keyword in label_lower for keyword in ["floor", "ground", "surface"]):
        return "floor"
    
    # Ceiling classification priority
    if any(keyword in label_lower for keyword in ["ceiling", "roof", "top"]):
        return "ceiling"
    
    # Dynamic environment classification (doors, corners, entrances, etc.)
    dynamic_keywords = ["door", "corner", "edge", "entrance", "exit", "opening", "passage", "intersection", "junction", "turning", "column", "pillar", "beam"]
    if any(keyword in label_lower for keyword in dynamic_keywords):
        return "dynamic"
    
    # Classify remaining categories
    for category, info in SAFETY_CATEGORIES.items():
        if category in ["floor", "ceiling", "dynamic"]:
            continue  # Already handled above
        
        for obj_keyword in info["objects"]:
            if obj_keyword in label_lower or label_lower in obj_keyword:
                return category
    
    # Default to safe classification
    return "safe"

def create_colored_mask_overlay(image, masks, labels, confidences, boxes):
    """Create mask overlay with transparent colors (spatial color classification + risk visualization)"""
    overlay = image.copy()
    
    print(f"Creating risk-based visualization for {len(masks)} objects...")
    
    # Calculate risk scores
    risk_scores = []
    risk_levels = []
    
    for i, (mask, label, conf, box) in enumerate(zip(masks, labels, confidences, boxes)):
        # Safety category classification
        category = categorize_object(label)
        
        # Risk score calculation
        risk_score = calculate_risk_score(category, conf, box, image.shape)
        risk_level = get_risk_level(risk_score)
        
        risk_scores.append(risk_score)
        risk_levels.append(risk_level)
        
        # Risk-based visualization parameters
        viz_params = get_risk_visualization_params(risk_score, risk_level)
        
        # Base category color
        base_color = SAFETY_CATEGORIES[category]["color"]
        
        # Apply color to mask area
        mask_bool = mask[0].astype(bool)
        overlay[mask_bool] = overlay[mask_bool] * (1 - viz_params["alpha"]) + np.array(base_color) * viz_params["alpha"]
        
        # Add risk-based border
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), viz_params["risk_color"], viz_params["border_thickness"])
        
        # Add label text (English only, no emojis)
        category_names = {
            "floor": "FLOOR",
            "ceiling": "CEILING", 
            "safe": "SAFE",
            "dynamic": "DYNAMIC",
            "caution": "CAUTION",
            "danger": "DANGER"
        }
        
        risk_names = {
            "SAFE": "SAFE",
            "LOW": "LOW",
            "MEDIUM": "MED",
            "HIGH": "HIGH",
            "CRITICAL": "CRIT"
        }
        
        category_name = category_names.get(category, "UNKNOWN")
        risk_name = risk_names.get(risk_level, "UNKNOWN")
        
        # Text composition (English only, no emojis)
        text = f"{category_name}: {label}"
        risk_text = f"{risk_name} ({risk_score:.1f})"
        
        # Text color based on category
        if category in ["danger", "floor"]:
            text_color = (255, 255, 255)  # White
        else:
            text_color = (0, 0, 0)  # Black
        
        # Text background (color adjusted based on risk)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        (risk_width, risk_height), _ = cv2.getTextSize(risk_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        max_width = max(text_width, risk_width)
        total_height = text_height + risk_height + 20
        
        # Background color (based on risk)
        bg_color = viz_params["risk_color"]
        
        # Background rectangle
        cv2.rectangle(overlay, (x1, y1-total_height-10), (x1+max_width+20, y1), bg_color, -1)
        
        # Draw text
        cv2.putText(overlay, text, (x1+5, y1-risk_height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(overlay, risk_text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    
    return overlay, risk_scores, risk_levels

def main():
    print("Risk-based Office Safety Analysis Starting...")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load models
    print("Loading Models...")
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    grounding_model = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT, device=device)
    
    # Image preprocessing and loading
    print("Preprocessing and Loading Image...")
    preprocessed_image = preprocess_image(IMG_PATH)
    image_source, image = load_image(IMG_PATH)
    
    # Set image to SAM2 predictor
    sam2_predictor.set_image(image_source)
    
    # Multi-threshold object detection
    boxes, confidences, labels = multi_threshold_detection(grounding_model, image, device)
    
    if len(boxes) == 0:
        print("No objects detected.")
        return
    
    # Box coordinate transformation
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
    
    # Apply NMS
    input_boxes, confidences, labels = apply_nms(input_boxes, confidences, labels)
    
    # Filter low confidence
    input_boxes, confidences, labels = filter_low_confidence(input_boxes, confidences, labels)
    
    if len(input_boxes) == 0:
        print("No objects detected after filtering.")
        return
    
    # Convert to numpy
    input_boxes = input_boxes.numpy()
    
    # Segmentation with SAM2
    print("Performing Segmentation...")
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    # Result visualization (risk-based)
    print("Creating Risk-based Visualization...")
    result_image, risk_scores, risk_levels = create_colored_mask_overlay(
        image_source, masks, labels, confidences, input_boxes
    )
    
    # Save results
    output_dir = "/root/ws/src/risk_nav/src/sample/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save risk analysis result image
    output_path = os.path.join(output_dir, "office_risk_assessment.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    
    # Save analysis results JSON
    results = {
        "image_path": IMG_PATH,
        "total_objects": len(labels),
        "detection_settings": {
            "multi_threshold": True,
            "nms_applied": True,
            "confidence_filtering": True,
            "image_preprocessing": True,
            "spatial_categorization": True,
            "risk_assessment": True
        },
        "risk_summary": {
            "total_risk_score": float(np.mean(risk_scores)),
            "risk_distribution": {
                "SAFE": 0,
                "LOW": 0,
                "MEDIUM": 0,
                "HIGH": 0,
                "CRITICAL": 0
            },
            "highest_risk_object": None,
            "lowest_risk_object": None
        },
        "spatial_summary": {
            "floor_objects": [],
            "ceiling_objects": [],
            "safe_objects": [],
            "dynamic_objects": [],
            "caution_objects": [],
            "danger_objects": []
        },
        "detailed_results": []
    }
    
    # Calculate risk distribution
    for level in risk_levels:
        results["risk_summary"]["risk_distribution"][level] += 1
    
    # Find highest/lowest risk objects
    if risk_scores:
        max_risk_idx = np.argmax(risk_scores)
        min_risk_idx = np.argmin(risk_scores)
        
        results["risk_summary"]["highest_risk_object"] = {
            "label": labels[max_risk_idx],
            "risk_score": float(risk_scores[max_risk_idx]),
            "risk_level": risk_levels[max_risk_idx]
        }
        
        results["risk_summary"]["lowest_risk_object"] = {
            "label": labels[min_risk_idx],
            "risk_score": float(risk_scores[min_risk_idx]),
            "risk_level": risk_levels[min_risk_idx]
        }
    
    for i, (box, conf, label, mask, risk_score, risk_level) in enumerate(zip(input_boxes, confidences, labels, masks, risk_scores, risk_levels)):
        category = categorize_object(label)
        
        # Spatial category classification
        results["spatial_summary"][f"{category}_objects"].append({
            "label": label,
            "confidence": float(conf),
            "bbox": box.tolist(),
            "category_description": SAFETY_CATEGORIES[category]["description"],
            "risk_score": float(risk_score),
            "risk_level": risk_level
        })
        
        # Encode mask to RLE format
        mask_uint8 = (mask[0] * 255).astype(np.uint8)
        mask_rle = mask_util.encode(np.asfortranarray(mask_uint8))
        mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
        
        results["detailed_results"].append({
            "id": i,
            "label": label,
            "confidence": float(conf),
            "spatial_category": category,
            "category_description": SAFETY_CATEGORIES[category]["description"],
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "risk_description": RISK_LEVELS[risk_level]["description"],
            "bbox": box.tolist(),
            "mask": mask_rle
        })
    
    # Save JSON file
    json_path = os.path.join(output_dir, "office_risk_assessment.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print result summary
    print("\n" + "="*70)
    print("RISK-BASED OFFICE SAFETY ANALYSIS RESULTS")
    print("="*70)
    print(f"Total detected objects: {len(labels)}")
    print(f"Average risk score: {np.mean(risk_scores):.1f}/100")
    
    # Print risk distribution
    print("\nRisk distribution:")
    for level, count in results["risk_summary"]["risk_distribution"].items():
        percentage = (count / len(labels)) * 100
        print(f"  {RISK_LEVELS[level]['description']}: {count} objects ({percentage:.1f}%)")
    
    # Print highest/lowest risk objects
    if results["risk_summary"]["highest_risk_object"]:
        highest = results["risk_summary"]["highest_risk_object"]
        print(f"\nHighest risk object: {highest['label']} ({highest['risk_score']:.1f} points, {highest['risk_level']})")
    
    if results["risk_summary"]["lowest_risk_object"]:
        lowest = results["risk_summary"]["lowest_risk_object"]
        print(f"Lowest risk object: {lowest['label']} ({lowest['risk_score']:.1f} points, {lowest['risk_level']})")
    
    print("\nObjects detected by spatial category (with risk levels):")
    
    # Print spatial results
    spatial_order = ["floor", "ceiling", "safe", "dynamic", "caution", "danger"]
    spatial_names = {
        "floor": "FLOOR",
        "ceiling": "CEILING",
        "safe": "SAFE",
        "dynamic": "DYNAMIC",
        "caution": "CAUTION",
        "danger": "DANGER"
    }
    
    for category in spatial_order:
        objects = results["spatial_summary"][f"{category}_objects"]
        if objects:
            name = spatial_names[category]
            description = SAFETY_CATEGORIES[category]["description"]
            print(f"\n{name} ({description}):")
            for obj in objects:
                print(f"  - {obj['label']} (confidence: {obj['confidence']:.2f}, risk: {obj['risk_score']:.1f}/{obj['risk_level']})")
    
    print(f"\nResults saved:")
    print(f"  Risk visualization: {output_path}")
    print(f"  Detailed JSON: {json_path}")
    print("="*70)

if __name__ == "__main__":
    main() 