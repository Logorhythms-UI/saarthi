"""
Zero-shot object detection benchmark tool.
"""
import asyncio
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import torch
from nicegui import ui
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import YOLO

# Global models dictionary
models_cache = {}

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_models():
    if models_cache:
        return
    device = get_device()
    
    print("Loading Grounding DINO...")
    models_cache["dino_processor"] = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    models_cache["dino_model"] = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)
    
    print("Loading OWL-ViT...")
    models_cache["owl_processor"] = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
    models_cache["owl_model"] = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    
    print("Loading YOLO-World...")
    models_cache["yolo_model"] = YOLO("yolov8s-world.pt")
    
    print("Loading LLMDet...")
    models_cache["llmdet_processor"] = AutoProcessor.from_pretrained("iSEE-Laboratory/llmdet_tiny")
    models_cache["llmdet_model"] = AutoModelForZeroShotObjectDetection.from_pretrained("iSEE-Laboratory/llmdet_tiny").to(device)
    print("All models loaded.")

def draw_boxes(image: Image.Image, boxes: list[list[float]], color="red") -> Image.Image:
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    for box in boxes:
        draw.rectangle(box, outline=color, width=3)
    return img_copy

def infer_transformers(model_key: str, processor_key: str, image: Image.Image, query: str) -> dict[str, Any]:
    processor = models_cache[processor_key]
    model = models_cache[model_key]
    device = get_device()
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
    else:
        # Start tracking CPU RAM
        tracemalloc.start()

    start_time = time.time()
    inputs = processor(images=image, text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]
    inference_time = time.time() - start_time

    memory_mb = 0.0
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated()
        memory_mb = (peak_mem - start_mem) / (1024 * 1024)
    else:
        # Get peak memory usage during the traced block for CPU
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_mb = peak_mem / (1024 * 1024)
    
    boxes = results["boxes"].cpu().numpy().tolist()
    scores = results["scores"].cpu().numpy().tolist()
    
    if scores:
        best_idx = scores.index(max(scores))
        best_box = [boxes[best_idx]]
        best_score = scores[best_idx] * 100
    else:
        best_box = []
        best_score = 0.0
        
    return {
        "boxes": best_box,
        "score": best_score,
        "inference_time": inference_time,
        "memory_mb": memory_mb, # ADDED
        "failure": len(best_box) == 0
    }

def infer_yolo(image: Image.Image, query: str) -> dict[str, Any]:
    model = models_cache["yolo_model"]
    device = get_device()
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
    else:
        # Start tracking CPU RAM
        tracemalloc.start()

    start_time = time.time()
    model.set_classes([query])
    results = model.predict(image, verbose=False)
    inference_time = time.time() - start_time
    
    memory_mb = 0.0
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated()
        memory_mb = (peak_mem - start_mem) / (1024 * 1024)
    else:
        # Get peak memory usage during the traced block for CPU
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_mb = peak_mem / (1024 * 1024)
    
    boxes = []
    best_score = 0.0
    if len(results) > 0 and len(results[0].boxes) > 0:
        box_data = results[0].boxes[0]
        boxes = box_data.xyxy.cpu().numpy().tolist()
        best_score = float(box_data.conf.cpu().numpy()[0]) * 100
        
    return {
        "boxes": boxes,
        "score": best_score,
        "inference_time": inference_time,
        "memory_mb": memory_mb, # ADDED
        "failure": len(boxes) == 0
    }

def process_dino(image: Image.Image, query: str) -> dict[str, Any]:
    # DINO expects query to end with .
    norm_query = query.strip()
    if not norm_query.endswith("."):
        norm_query += "."
    return infer_transformers("dino_model", "dino_processor", image, norm_query)

def process_owl(image: Image.Image, query: str) -> dict[str, Any]:
    return infer_transformers("owl_model", "owl_processor", image, query)

def process_llmdet(image: Image.Image, query: str) -> dict[str, Any]:
    return infer_transformers("llmdet_model", "llmdet_processor", image, query)

def process_yolo(image: Image.Image, query: str) -> dict[str, Any]:
    return infer_yolo(image, query)


class BenchmarkApp:
    def __init__(self):
        self.image_dir = Path(__file__).parent / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.images = sorted([f for f in self.image_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        self.current_index = 0
        
        # We output to the project root
        self.results_file = Path(__file__).parent.parent.parent / "results.json"
        if not self.results_file.exists():
            with open(self.results_file, "w") as f:
                json.dump([], f)
                
        self.current_results = {}
        self.feedbacks_given = 0
        
    def save_metric(self, model_name: str, query: str, image_path: str, is_correct: bool, result_data: dict[str, Any]):
        try:
            with open(self.results_file) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
            
        metric = {
            "model": model_name,
            "query": query,
            "image_path": str(image_path),
            "success": 1 if is_correct else 0,
            "false_positive": 0 if is_correct else (0 if result_data["failure"] else 1),
            "failure": 1 if result_data["failure"] else 0,
            "confidence_score": result_data["score"],
            "inference_time": result_data["inference_time"],
            "memory_footprint_mb": result_data["memory_mb"] # ADDED
        }
        data.append(metric)
        with open(self.results_file, "w") as f:
            json.dump(data, f, indent=4)

    def handle_feedback(self, model_name: str, query: str, image_path: str, is_correct: bool, result_data: dict[str, Any], container: ui.column):
        self.save_metric(model_name, query, image_path, is_correct, result_data)
        container.clear()
        with container:
            ui.label("Feedback recorded").classes("text-green-500 font-bold")

    @ui.refreshable
    def refresh_ui(self):
        ui.label("Zero-Shot Object Detection Benchmark").classes("text-2xl font-bold mb-4")
        
        if not self.images:
            ui.label("No images found in images/ directory. Please add some and refresh.").classes("text-red-500")
            return
            
        current_image = self.images[self.current_index]
        ui.label(f"Image {self.current_index + 1} of {len(self.images)}: {current_image.name}")
        
        # Display main image
        ui.image(current_image).style("max-height: 400px; max-width: 100%; object-fit: contain;")
        
        query_input = ui.input(label="Object to detect")
        
        results_container = ui.row().classes("w-full gap-4 mt-8 flex-wrap justify-center")
        loader = ui.spinner(size='lg').classes("mt-4")
        loader.set_visibility(False)
        
        async def run_inference():
            query = query_input.value
            if not query:
                ui.notify("Please enter a query")
                return
                
            self.feedbacks_given = 0
            loader.set_visibility(True)
            results_container.clear()
            
            # Load image
            img = Image.open(current_image).convert("RGB")
            
            # Run models concurrently
            try:
                res_dino = await asyncio.to_thread(process_dino, img, query)
                res_owl = await asyncio.to_thread(process_owl, img, query)
                res_yolo = await asyncio.to_thread(process_yolo, img, query)
                res_llmdet = await asyncio.to_thread(process_llmdet, img, query)
            except (RuntimeError, OSError, ValueError) as e:
                ui.notify(f"Error during inference: {e}")
                loader.set_visibility(False)
                return
            
            # Display results
            try:
                res_dino, res_owl, res_yolo, res_llmdet = await asyncio.gather(*tasks)
            except (RuntimeError, OSError, ValueError) as e:
                ui.notify(f"Error during inference: {e}")
                loader.set_visibility(False)
                return
                
            loader.set_visibility(False)
            
            model_results = [
                ("Grounding DINO", res_dino),
                ("OWL-ViT", res_owl),
                ("YOLO-World", res_yolo),
                ("LLMDet", res_llmdet)
            ]
            
            with results_container:
                for name, res in model_results:
                    with ui.card().classes("items-center w-64"):
                        ui.label(name).classes("font-bold")
                        ui.label(f"Time: {res['inference_time']:.2f}s | Mem: {res['memory_mb']:.1f} MB | Conf: {res['score']:.1f}%").classes("text-xs")
                        
                        if res["failure"]:
                            ui.label("No object found").classes("text-red-500 my-4")
                        else:
                            annotated_img = draw_boxes(img, res["boxes"])
                            # Save temporary annotated image
                            out_path = self.image_dir / f"temp_{name.replace(' ', '_')}.jpg"
                            annotated_img.save(out_path)
                            # add cache bypass query string to update image correctly if it exists
                            ui.image(f"{out_path}?t={time.time()}").style("max-height: 200px; max-width: 100%; object-fit: contain;")
                            
                        # Feedback buttons
                        btn_container = ui.column().classes("w-full items-center mt-2")
                        with btn_container, ui.row():
                                ui.button("Correct", color="green", on_click=lambda n=name, q=query, p=current_image, r=res, c=btn_container: self.handle_feedback(n, q, p, True, r, c))
                                ui.button("Incorrect", color="red", on_click=lambda n=name, q=query, p=current_image, r=res, c=btn_container: self.handle_feedback(n, q, p, False, r, c))
                                
        ui.button("Run Models", on_click=run_inference).classes("mt-4")
        
        with ui.row().classes("mt-8"):
            ui.button("Previous Image", on_click=self.prev_image)
            ui.button("Next Image", on_click=self.next_image)

    def next_image(self):
        if self.images:
            self.current_index = (self.current_index + 1) % len(self.images)
            self.refresh_ui()
            
    def prev_image(self):
        if self.images:
            self.current_index = (self.current_index - 1) % len(self.images)
            self.refresh_ui()

def main():
    print("Pre-loading models (this may take a while)...")
    try:
        load_models()
    except (RuntimeError, OSError, ValueError) as e:
        print(f"Warning: Model loading error: {e}")
        
    app = BenchmarkApp()
    app.refresh_ui()
    
    ui.run(title="Zero-Shot Benchmarking", port=8081)

if __name__ in {"__main__", "__mp_main__"}:
    main()
