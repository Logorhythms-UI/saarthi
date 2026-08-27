"""
Zero-shot object detection benchmark tool.
"""
import asyncio
import base64
import json
import time
import tracemalloc
from io import BytesIO
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

def pil_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    # Downscale for UI if very large to save DOM memory
    if img.width > 800 or img.height > 800:
        img.thumbnail((800, 800))
    img.save(buffer, format="JPEG", quality=75)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

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
    
    if hasattr(processor, "post_process_grounded_object_detection"):
        results = processor.post_process_grounded_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]
    elif hasattr(processor, "post_process_object_detection"):
        results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]
    else:
        results = processor.post_process(outputs, target_sizes=target_sizes)[0]
        
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
        "memory_mb": memory_mb,
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
        "memory_mb": memory_mb,
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
        all_images = sorted([f for f in self.image_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        self.images = all_images
        self.results_file = Path(__file__).parent.parent.parent / "results.json"
        
        self.phase = 1
        self.current_index = 0
        
        self.queries = {} # image_path -> query
        self.inference_results = {} # image_path -> { model_name: result_dict }
        self.feedbacks = {} # image_path -> { model_name: is_correct }
        
        self.background_tasks = set()
        self.semaphore = asyncio.Semaphore(1) # process 1 image at a time to save memory
        self.processed_count = 0
        
        self.query_inputs = {}
        
    async def run_inference_background(self, img_path: Path, query: str):
        async with self.semaphore:
            img = Image.open(img_path).convert("RGB")
            try:
                res_dino, res_owl, res_yolo, res_llmdet = await asyncio.gather(
                    asyncio.to_thread(process_dino, img, query),
                    asyncio.to_thread(process_owl, img, query),
                    asyncio.to_thread(process_yolo, img, query),
                    asyncio.to_thread(process_llmdet, img, query)
                )
                self.inference_results[str(img_path)] = {
                    "Grounding DINO": res_dino,
                    "OWL-ViT": res_owl,
                    "YOLO-World": res_yolo,
                    "LLMDet": res_llmdet
                }
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                self.inference_results[str(img_path)] = None
            finally:
                self.processed_count += 1

    def start_phase_2(self):
        self.phase = 2
        self.refresh_ui()
            
    def record_feedback(self, img_path: str, model_name: str, is_correct: bool, btn_container: ui.row):
        if img_path not in self.feedbacks:
            self.feedbacks[img_path] = {}
        self.feedbacks[img_path][model_name] = is_correct
        btn_container.clear()
        with btn_container:
            ui.label("Correct" if is_correct else "Incorrect").classes("font-bold text-lg " + ("text-green-600" if is_correct else "text-red-600"))

    def submit_phase_2(self):
        # Validate that all feedbacks are provided for successful inferences
        for img_path in self.images:
            path_str = str(img_path)
            res = self.inference_results.get(path_str)
            if res:
                if path_str not in self.feedbacks or len(self.feedbacks[path_str]) < len(res):
                    ui.notify("Please provide feedback (Correct/Incorrect) for all models on all images.", type="warning")
                    return
        self.finish_benchmark()
            
    def finish_benchmark(self):
        self.phase = 3
        # calculate metrics
        output = []
        model_stats = {m: {"conf": [], "time": [], "mem": []} for m in ["Grounding DINO", "OWL-ViT", "YOLO-World", "LLMDet"]}
        
        for img_path in self.images:
            path_str = str(img_path)
            res = self.inference_results.get(path_str)
            if not res: continue
            
            for m_name, m_res in res.items():
                is_correct = self.feedbacks.get(path_str, {}).get(m_name, False)
                metric = {
                    "model": m_name,
                    "query": self.queries.get(path_str, ""),
                    "image_path": path_str,
                    "success": 1 if is_correct else 0,
                    "false_positive": 0 if is_correct else (0 if m_res["failure"] else 1),
                    "failure": 1 if m_res["failure"] else 0,
                    "confidence_score": m_res["score"],
                    "inference_time": m_res["inference_time"],
                    "memory_footprint_mb": m_res["memory_mb"]
                }
                output.append(metric)
                
                model_stats[m_name]["conf"].append(m_res["score"])
                model_stats[m_name]["time"].append(m_res["inference_time"])
                model_stats[m_name]["mem"].append(m_res["memory_mb"])
                
        with open(self.results_file, "w") as f:
            json.dump(output, f, indent=4)
            
        self.summary_stats = {}
        for m, stats in model_stats.items():
            if not stats["conf"]: continue
            self.summary_stats[m] = {
                "avg_conf": sum(stats["conf"]) / len(stats["conf"]),
                "max_conf": max(stats["conf"]),
                "min_conf": min(stats["conf"]),
                "avg_time": sum(stats["time"]) / len(stats["time"]),
                "max_time": max(stats["time"]),
                "min_time": min(stats["time"]),
                "avg_mem": sum(stats["mem"]) / len(stats["mem"])
            }
            
        self.refresh_ui()

    @ui.refreshable
    def refresh_ui(self):
        with ui.column().classes("w-full max-w-6xl mx-auto p-6 items-stretch"):
            ui.label("Zero-Shot Object Detection Benchmark").classes("text-4xl font-extrabold mb-8 text-center text-indigo-700")
            
            if not self.images:
                ui.label("No images found in images/ directory. Please add some and refresh.").classes("text-red-500 text-xl text-center")
                return
                
            if self.phase == 1:
                with ui.column().classes("w-full bg-white border border-gray-200 rounded-xl p-8 mb-8"):
                    ui.label("Phase 1: Input Queries").classes("text-3xl font-extrabold mb-2 text-gray-800")
                    ui.label("Label all images below and submit them at once.").classes("text-lg mb-6 text-gray-500")
                    
                    self.query_inputs = {}
                    
                    with ui.column().classes("w-full gap-6 items-stretch"):
                        for img_path in self.images:
                            with ui.row().classes("w-full flex-row items-center gap-6 p-4 rounded-xl bg-gray-50 border border-gray-200"):
                                ui.image(img_path).classes("w-64 rounded-lg").style("object-fit: contain; max-height: 200px;")
                                self.query_inputs[str(img_path)] = ui.input(label="Object to detect", placeholder="e.g. a red car").classes("flex-grow text-xl")
                                
                    def on_submit_all():
                        for p, inp in self.query_inputs.items():
                            if not inp.value.strip():
                                ui.notify("Please enter a query for all images before submitting.", type="warning")
                                return
                                
                        for p, inp in self.query_inputs.items():
                            query = inp.value.strip()
                            self.queries[p] = query
                            task = asyncio.create_task(self.run_inference_background(Path(p), query))
                            self.background_tasks.add(task)
                            task.add_done_callback(self.background_tasks.discard)
                            
                        self.phase = 1.5
                        self.refresh_ui()

                    ui.button("Submit All Queries", on_click=on_submit_all).classes("mt-8 w-full bg-indigo-600 text-white text-xl py-3 rounded-lg font-bold hover:bg-indigo-700 cursor-pointer")
                
            elif self.phase == 1.5:
                with ui.column().classes("w-full bg-white border border-gray-200 rounded-xl p-16 text-center"):
                    ui.label("Running Inference Models").classes("text-3xl font-bold mb-4 text-indigo-700")
                    ui.spinner('dots', size='xl').classes("mx-auto mb-6 text-indigo-500")
                    progress_label = ui.label(f"Processed: {self.processed_count} / {len(self.images)} images").classes("text-xl text-gray-600 font-medium")
                    
                    def check_progress():
                        if getattr(self, "phase", 1.5) != 1.5:
                            return
                        progress_label.set_text(f"Processed: {self.processed_count} / {len(self.images)} images")
                        if self.processed_count >= len(self.images):
                            if hasattr(self, 'timer'):
                                self.timer.deactivate()
                            self.start_phase_2()
                            
                    self.timer = ui.timer(0.5, check_progress)
                    
            elif self.phase == 2:
                with ui.row().classes("w-full justify-between items-center mb-6"):
                    with ui.column():
                        ui.label("Phase 2: Review Results").classes("text-3xl font-extrabold text-gray-800")
                        ui.label("Review the detections for all images below and mark them Correct or Incorrect.").classes("text-lg text-gray-500 mt-2")
                    ui.button("Finish & View Summary", on_click=self.submit_phase_2).classes("bg-indigo-600 text-white text-xl py-2 px-6 rounded-lg font-bold hover:bg-indigo-700 cursor-pointer")

                with ui.column().classes("w-full gap-8 items-stretch"):
                    for idx, current_image in enumerate(self.images):
                        path_str = str(current_image)
                        res = self.inference_results.get(path_str)
                        
                        with ui.column().classes("w-full bg-white border border-gray-200 rounded-xl p-6"):
                            ui.label(f"Image {idx + 1} | Query: '{self.queries.get(path_str, 'N/A')}'").classes("text-2xl font-bold text-indigo-600 mb-6")
                            
                            if not res:
                                ui.label("Inference failed for this image.").classes("text-xl text-red-500 font-bold")
                                continue
                                
                            img = Image.open(current_image).convert("RGB")
                            
                            with ui.grid(columns=2).classes("w-full gap-6"):
                                for m_name, m_res in res.items():
                                    with ui.column().classes("items-center w-full bg-gray-50 rounded-lg border border-gray-200 overflow-hidden"):
                                        ui.label(m_name).classes("w-full text-center py-2 bg-gray-100 font-bold text-lg border-b border-gray-200")
                                        
                                        with ui.column().classes("p-4 w-full items-center"):
                                            ui.label(f"Time: {m_res['inference_time']:.2f}s | Mem: {m_res['memory_mb']:.1f} MB").classes("text-sm text-gray-500 mb-1")
                                            ui.label(f"Confidence: {m_res['score']:.1f}%").classes("text-sm font-semibold text-gray-700 mb-4")
                                            
                                            if m_res["failure"]:
                                                ui.label("No object found").classes("text-red-500 font-bold my-10")
                                            else:
                                                annotated = draw_boxes(img, m_res["boxes"])
                                                b64_str = pil_to_base64(annotated)
                                                ui.image(b64_str).classes("rounded-lg w-full").style("object-fit: contain; max-height: 250px;")
                                                
                                            ui.separator().classes("w-full my-4")
                                            
                                            if path_str in self.feedbacks and m_name in self.feedbacks[path_str]:
                                                is_correct = self.feedbacks[path_str][m_name]
                                                ui.label("Correct" if is_correct else "Incorrect").classes("font-bold text-lg " + ("text-green-600" if is_correct else "text-red-600"))
                                            else:
                                                btn_container = ui.row().classes("w-full justify-center gap-4 mt-2")
                                                with btn_container:
                                                    ui.button("Correct", color="positive", icon="check", on_click=lambda p=path_str, m=m_name, c=btn_container: self.record_feedback(p, m, True, c)).classes("rounded-full px-4")
                                                    ui.button("Incorrect", color="negative", icon="close", on_click=lambda p=path_str, m=m_name, c=btn_container: self.record_feedback(p, m, False, c)).classes("rounded-full px-4")
                                                    
                ui.button("Finish & View Summary", on_click=self.submit_phase_2).classes("mt-10 mx-auto bg-indigo-600 text-white text-xl py-3 px-12 rounded-lg font-bold hover:bg-indigo-700 block")
                
            elif self.phase == 3:
                ui.label("Benchmark Complete!").classes("text-4xl font-extrabold text-center text-green-600 mb-2")
                ui.label(f"Detailed results saved to {self.results_file}").classes("text-center text-gray-500 mb-8")
                
                if not hasattr(self, 'summary_stats') or not self.summary_stats:
                    return
                    
                for m_name, s in self.summary_stats.items():
                    with ui.column().classes("w-full bg-white border border-gray-200 rounded-xl p-8 mb-8"):
                        ui.label(m_name).classes("text-3xl font-bold mb-6 text-gray-800 border-b border-gray-200 pb-2")
                        
                        correct_count = sum(1 for img_path in self.images if self.feedbacks.get(str(img_path), {}).get(m_name, False))
                        accuracy = (correct_count / len(self.images)) * 100 if self.images else 0
                        
                        with ui.row().classes("w-full gap-6 flex-wrap justify-center"):
                            # Accuracy
                            with ui.column().classes("flex-1 min-w-[220px] rounded-xl p-5 text-center border items-center justify-center shadow-sm bg-green-50 border-green-200"):
                                ui.icon('check_circle', color='green', size='md')
                                ui.label("Accuracy").classes("text-sm font-semibold text-gray-600 mt-2")
                                ui.label(f"{accuracy:.1f}%").classes("text-3xl font-bold mt-1 text-green-700")
                            
                            # Total Evaluated
                            with ui.column().classes("flex-1 min-w-[220px] rounded-xl p-5 text-center border items-center justify-center shadow-sm bg-blue-50 border-blue-200"):
                                ui.icon('image', color='blue', size='md')
                                ui.label("Total Evaluated").classes("text-sm font-semibold text-gray-600 mt-2")
                                ui.label(f"{len(self.images)}").classes("text-3xl font-bold mt-1 text-blue-700")
                                
                            # Memory Footprint
                            with ui.column().classes("flex-1 min-w-[220px] rounded-xl p-5 text-center border items-center justify-center shadow-sm bg-purple-50 border-purple-200"):
                                ui.icon('memory', color='purple', size='md')
                                ui.label("Memory Footprint").classes("text-sm font-semibold text-gray-600 mt-2")
                                ui.label(f"{s['avg_mem']:.2f} MB").classes("text-3xl font-bold mt-1 text-purple-700")
                                
                            # Hardware Used
                            with ui.column().classes("flex-1 min-w-[220px] rounded-xl p-5 text-center border items-center justify-center shadow-sm bg-gray-50 border-gray-200"):
                                ui.icon('developer_board', color='gray', size='md')
                                ui.label("Hardware Used").classes("text-sm font-semibold text-gray-600 mt-2")
                                ui.label(f"{get_device().upper()}").classes("text-3xl font-bold mt-1 text-gray-700")
                                
                        with ui.row().classes("w-full gap-6 flex-wrap justify-center mt-6"):
                            # Inference Time
                            with ui.column().classes("flex-1 min-w-[220px] rounded-xl p-5 text-center border items-center justify-center shadow-sm bg-orange-50 border-orange-200"):
                                ui.icon('timer', color='orange', size='md')
                                ui.label("Avg Inference Time").classes("text-sm font-semibold text-gray-600 mt-2")
                                ui.label(f"{s['avg_time']:.4f}s").classes("text-3xl font-bold mt-1 text-orange-700")
                                
                            # Avg Confidence
                            with ui.column().classes("flex-1 min-w-[220px] rounded-xl p-5 text-center border items-center justify-center shadow-sm bg-indigo-50 border-indigo-200"):
                                ui.icon('analytics', color='indigo', size='md')
                                ui.label("Avg Confidence").classes("text-sm font-semibold text-gray-600 mt-2")
                                ui.label(f"{s['avg_conf']/100:.3f}").classes("text-3xl font-bold mt-1 text-indigo-700")
                                
                            # Avg Power
                            with ui.column().classes("flex-1 min-w-[220px] rounded-xl p-5 text-center border items-center justify-center shadow-sm bg-yellow-50 border-yellow-200"):
                                ui.icon('bolt', color='orange', size='md')
                                ui.label("Avg Power").classes("text-sm font-semibold text-gray-600 mt-2")
                                ui.label("N/A").classes("text-3xl font-bold mt-1 text-orange-700")

def main():
    print("Pre-loading models (this may take a while)...")
    try:
        load_models()
    except Exception as e:
        print(f"Warning: Model loading error: {e}")
        
    app = BenchmarkApp()
    app.refresh_ui()
    
    ui.run(title="Zero-Shot Benchmarking", port=8081)

if __name__ in {"__main__", "__mp_main__"}:
    main()
