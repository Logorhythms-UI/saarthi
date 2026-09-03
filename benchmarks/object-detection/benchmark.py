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


# ─── Shared state file helpers ───────────────────────────────────────────────

STATE_FILE = Path(__file__).parent / "benchmark_state.json"

def save_state(state_dict: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state_dict, f)

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Could not load state: {e}")
    return {}


# ─── Shared app state (single-process, survives page reloads) ────────────────

class AppState:
    """Holds all mutable benchmark state in memory; serialised to disk on every change."""

    def __init__(self, images: list[Path]):
        self.images = images
        self.results_file = Path(__file__).parent.parent.parent / "results.json"

        # Core dictionaries – keyed by image *filename* (e.g. "img123.jpg")
        self.queries: dict[str, str] = {}
        self.inference_results: dict[str, dict] = {}
        self.feedbacks: dict[str, dict] = {}

        # Navigation
        self.phase = 1
        self.current_page: int = 0

        # Concurrency bookkeeping (not persisted)
        self.background_tasks: set = set()
        self.active_tasks: set = set()
        self.semaphore = asyncio.Semaphore(1)
        self.query_inputs: dict = {}

        # ── Restore from disk ──
        disk = load_state()
        if disk:
            self.queries = disk.get("queries", {})
            self.inference_results = disk.get("inference_results", {})
            self.feedbacks = disk.get("feedbacks", {})
            self.phase = disk.get("phase", 1)
            self.current_page = disk.get("current_page", 0)
            print(f"Loaded {len(self.queries)} queries, {len(self.inference_results)} inferences, {len(self.feedbacks)} feedbacks from state.")

        # Auto-jump to Phase 2 when all inferences are done
        if len(self.inference_results) >= len(self.images) and len(self.images) > 0 and self.phase < 2:
            self.phase = 2
            self._auto_page()

    def persist(self):
        save_state({
            "queries": self.queries,
            "inference_results": self.inference_results,
            "feedbacks": self.feedbacks,
            "phase": self.phase,
            "current_page": self.current_page,
        })

    def _auto_page(self):
        """Jump current_page to the first page that has un-rated images."""
        rated = 0
        for img_path in self.images:
            key = img_path.name
            res = self.inference_results.get(key)
            if res and len(self.feedbacks.get(key, {})) >= len(res):
                rated += 1
            else:
                break
        self.current_page = rated // 10
        self.persist()

    async def run_inference_background(self, img_path: Path, query: str):
        key = img_path.name
        self.active_tasks.add(key)
        async with self.semaphore:
            img = Image.open(img_path).convert("RGB")
            try:
                res_dino, res_owl, res_yolo, res_llmdet = await asyncio.gather(
                    asyncio.to_thread(process_dino, img, query),
                    asyncio.to_thread(process_owl, img, query),
                    asyncio.to_thread(process_yolo, img, query),
                    asyncio.to_thread(process_llmdet, img, query),
                )
                self.inference_results[key] = {
                    "Grounding DINO": res_dino,
                    "OWL-ViT": res_owl,
                    "YOLO-World": res_yolo,
                    "LLMDet": res_llmdet,
                }
            except (RuntimeError, OSError, ValueError) as e:
                print(f"Error processing {img_path}: {e}")
                self.inference_results[key] = None
            finally:
                self.persist()
                self.active_tasks.discard(key)

    def record_feedback(self, img_key: str, model_name: str, is_correct: bool):
        self.feedbacks.setdefault(img_key, {})[model_name] = is_correct
        self.persist()

    def compute_summary(self):
        output = []
        model_stats = {m: {"conf": [], "time": [], "mem": []} for m in ["Grounding DINO", "OWL-ViT", "YOLO-World", "LLMDet"]}

        for img_path in self.images:
            key = img_path.name
            res = self.inference_results.get(key)
            if not res:
                continue
            for m_name, m_res in res.items():
                is_correct = self.feedbacks.get(key, {}).get(m_name, False)
                metric = {
                    "model": m_name,
                    "query": self.queries.get(key, ""),
                    "image_path": str(img_path),
                    "success": 1 if is_correct else 0,
                    "false_positive": 0 if is_correct else (0 if m_res["failure"] else 1),
                    "failure": 1 if m_res["failure"] else 0,
                    "confidence_score": m_res["score"],
                    "inference_time": m_res["inference_time"],
                    "memory_footprint_mb": m_res["memory_mb"],
                }
                output.append(metric)
                model_stats[m_name]["conf"].append(m_res["score"])
                model_stats[m_name]["time"].append(m_res["inference_time"])
                model_stats[m_name]["mem"].append(m_res["memory_mb"])

        with open(self.results_file, "w") as f:
            json.dump(output, f, indent=4)

        summary = {}
        for m, stats in model_stats.items():
            if not stats["conf"]:
                continue
            summary[m] = {
                "avg_conf": sum(stats["conf"]) / len(stats["conf"]),
                "max_conf": max(stats["conf"]),
                "min_conf": min(stats["conf"]),
                "avg_time": sum(stats["time"]) / len(stats["time"]),
                "max_time": max(stats["time"]),
                "min_time": min(stats["time"]),
                "avg_mem": sum(stats["mem"]) / len(stats["mem"]),
            }
        return summary


# ─── Global app_state (initialised in main) ──────────────────────────────────
app_state: AppState = None  # type: ignore


# ─── UI Renderers ────────────────────────────────────────────────────────────

def render_phase1():
    st = app_state

    with ui.column().classes("w-full bg-white border border-gray-200 rounded-xl p-8 mb-8"):
        ui.label("Phase 1: Input Queries").classes("text-3xl font-extrabold mb-2 text-gray-800")
        ui.label("Label all images below and submit them at once.").classes("text-lg mb-6 text-gray-500")

        st.query_inputs = {}

        def handle_input_commit(key, inp):
            query = inp.value.strip()
            if not query:
                return
            if st.queries.get(key) == query and key in st.inference_results:
                return
            st.queries[key] = query
            st.persist()
            ui.notify(f"Saved & processing: '{query}'", type="info", position="bottom-right")
            img_path = next((img for img in st.images if img.name == key), None)
            if img_path:
                task = asyncio.create_task(st.run_inference_background(img_path, query))
                st.background_tasks.add(task)
                task.add_done_callback(st.background_tasks.discard)
            if len(st.queries) == len(st.images):
                ui.notify("All images labeled. Moving to next phase...", type="positive")
                on_submit_all()

        with ui.column().classes("w-full gap-6 items-stretch"):
            for img_path in st.images:
                key = img_path.name
                with ui.row().classes("w-full flex-row items-center gap-6 p-4 rounded-xl bg-gray-50 border border-gray-200"):
                    ui.image(img_path).classes("w-64 rounded-lg").style("object-fit: contain; max-height: 200px;")
                    inp = ui.input(label="Object to detect", placeholder="e.g. a red car").classes("flex-grow text-xl")
                    if key in st.queries:
                        inp.value = st.queries[key]
                    inp.on("blur", lambda e, k=key, i=inp: handle_input_commit(k, i))
                    inp.on("keydown.enter", lambda e, k=key, i=inp: handle_input_commit(k, i))
                    st.query_inputs[key] = inp

        def on_submit_all():
            missing = [k for k, inp in st.query_inputs.items() if not inp.value.strip()]
            if missing:
                ui.notify(f"Please enter a query for all images. ({len(missing)} remaining: {', '.join(missing[:3])})", type="warning", timeout=5000)
                return
            for k, inp in st.query_inputs.items():
                query = inp.value.strip()
                needs = (st.queries.get(k) != query) or (k not in st.inference_results and k not in st.active_tasks)
                if needs:
                    st.queries[k] = query
                    st.persist()
                    img_path = next((img for img in st.images if img.name == k), None)
                    if img_path:
                        task = asyncio.create_task(st.run_inference_background(img_path, query))
                        st.background_tasks.add(task)
                        task.add_done_callback(st.background_tasks.discard)
            if len(st.inference_results) < len(st.images) or len(st.background_tasks) > 0:
                st.phase = 1.5
                st.persist()
                ui.navigate.reload()
            else:
                st.phase = 2
                st._auto_page()
                ui.navigate.reload()

        ui.button("Finish & Review", on_click=on_submit_all).classes(
            "mt-8 w-full bg-indigo-600 text-white text-xl py-3 rounded-lg font-bold hover:bg-indigo-700 cursor-pointer"
        )


def render_phase1_5():
    st = app_state
    with ui.column().classes("w-full bg-white border border-gray-200 rounded-xl p-16 text-center"):
        ui.label("Running Inference Models").classes("text-3xl font-bold mb-4 text-indigo-700")
        ui.spinner("dots", size="xl").classes("mx-auto mb-6 text-indigo-500")
        progress_label = ui.label(f"Processed: {len(st.inference_results)} / {len(st.images)} images").classes("text-xl text-gray-600 font-medium")

        def check_progress():
            progress_label.set_text(f"Processed: {len(st.inference_results)} / {len(st.images)} images")
            if len(st.inference_results) >= len(st.images) and len(st.background_tasks) == 0:
                st.phase = 2
                st._auto_page()
                ui.navigate.reload()

        ui.timer(0.5, check_progress)


def render_phase2():
    st = app_state
    images_per_page = 10
    total_pages = max(1, (len(st.images) + images_per_page - 1) // images_per_page)
    start_idx = st.current_page * images_per_page
    end_idx = start_idx + images_per_page
    page_images = st.images[start_idx:end_idx]

    with ui.row().classes("w-full justify-between items-center mb-6"):
        with ui.column():
            ui.label(f"Phase 2: Review Results (Page {st.current_page + 1} of {total_pages})").classes("text-3xl font-extrabold text-gray-800")
            ui.label("Review the detections for all images below and mark them Correct or Incorrect.").classes("text-lg text-gray-500 mt-2")
        ui.button("Finish & View Summary", on_click=do_finish).classes(
            "bg-indigo-600 text-white text-xl py-2 px-6 rounded-lg font-bold hover:bg-indigo-700 cursor-pointer"
        )

    _render_pagination(total_pages)

    with ui.column().classes("w-full gap-8 items-stretch"):
        for idx, current_image in enumerate(page_images, start=start_idx):
            key = current_image.name
            res = st.inference_results.get(key)

            with ui.column().classes("w-full bg-white border border-gray-200 rounded-xl p-6"):
                ui.label(f"Image {idx + 1} | Query: '{st.queries.get(key, 'N/A')}'").classes("text-2xl font-bold text-indigo-600 mb-6")

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

                                existing = st.feedbacks.get(key, {}).get(m_name)
                                if existing is not None:
                                    label_text = "Correct" if existing else "Incorrect"
                                    label_color = "text-green-600" if existing else "text-red-600"
                                    ui.label(label_text).classes(f"font-bold text-lg {label_color}")
                                else:
                                    btn_container = ui.row().classes("w-full justify-center gap-4 mt-2")
                                    with btn_container:
                                        def make_cb(k, mn, correct, container):
                                            def cb():
                                                st.record_feedback(k, mn, correct)
                                                container.clear()
                                                with container:
                                                    lbl = "Correct" if correct else "Incorrect"
                                                    clr = "text-green-600" if correct else "text-red-600"
                                                    ui.label(lbl).classes(f"font-bold text-lg {clr}")
                                            return cb

                                        ui.button("Correct", color="positive", icon="check",
                                                  on_click=make_cb(key, m_name, True, btn_container)).classes("rounded-full px-4")
                                        ui.button("Incorrect", color="negative", icon="close",
                                                  on_click=make_cb(key, m_name, False, btn_container)).classes("rounded-full px-4")

    _render_pagination(total_pages)


def _render_pagination(total_pages: int):
    st = app_state
    with ui.row().classes("w-full justify-center gap-4 my-6"):
        if st.current_page > 0:
            ui.button("Previous Page", on_click=lambda: _go_page(st.current_page - 1)).classes("bg-gray-200 text-gray-800 font-bold px-6 py-3")
        ui.label(f"Page {st.current_page + 1} / {total_pages}").classes("text-lg font-semibold self-center")
        if st.current_page < total_pages - 1:
            ui.button("Next Page", on_click=lambda: _go_page(st.current_page + 1)).classes("bg-blue-600 text-white font-bold px-6 py-3")


def _go_page(page: int):
    app_state.current_page = page
    app_state.persist()
    ui.navigate.reload()


def do_finish():
    st = app_state
    images_per_page = 10
    missing_items = []
    for i, img_path in enumerate(st.images):
        key = img_path.name
        res = st.inference_results.get(key)
        if res:
            fb = st.feedbacks.get(key, {})
            missing_models = [m for m in res if m not in fb]
            if missing_models:
                page_num = (i // images_per_page) + 1
                missing_items.append((key, page_num, missing_models))

    if missing_items:
        with ui.dialog() as dlg, ui.card().classes("w-[600px] max-h-[80vh]"):
            ui.label("Missing Feedback").classes("text-2xl font-bold text-red-600 mb-4")
            ui.label(f"{len(missing_items)} image(s) still need your Correct/Incorrect response:").classes("text-lg text-gray-700 mb-4")
            with ui.scroll_area().classes("w-full max-h-[50vh]"):
                for img_name, page, models in missing_items:
                    with ui.row().classes("w-full items-center gap-2 py-2 border-b border-gray-100"):
                        ui.badge(f"Page {page}", color="blue").classes("text-sm")
                        ui.label(img_name).classes("font-mono text-sm flex-grow")
                        for m in models:
                            ui.badge(m, color="red").classes("text-xs")
            with ui.row().classes("w-full justify-end gap-4 mt-4"):
                first_page = missing_items[0][1] - 1
                ui.button("Go to first missing", on_click=lambda: (_go_page(first_page), dlg.close())).classes("bg-blue-600 text-white font-bold")
                ui.button("Close", on_click=dlg.close).classes("bg-gray-200 text-gray-800 font-bold")
        dlg.open()
        return

    st.phase = 3
    st.persist()
    ui.navigate.reload()


def render_phase3():
    st = app_state
    summary = st.compute_summary()

    ui.label("Benchmark Complete!").classes("text-4xl font-extrabold text-center text-green-600 mb-2")
    ui.label(f"Detailed results saved to {st.results_file}").classes("text-center text-gray-500 mb-8")

    if not summary:
        return

    for m_name, s in summary.items():
        with ui.column().classes("w-full bg-white border border-gray-200 rounded-xl p-8 mb-8"):
            ui.label(m_name).classes("text-3xl font-bold mb-6 text-gray-800 border-b border-gray-200 pb-2")

            correct_count = sum(1 for img in st.images if st.feedbacks.get(img.name, {}).get(m_name, False))
            accuracy = (correct_count / len(st.images)) * 100 if st.images else 0

            with ui.row().classes("w-full gap-6 flex-wrap justify-center"):
                with ui.column().classes("flex-1 min-w-[220px] rounded-xl p-5 text-center border items-center justify-center shadow-sm bg-green-50 border-green-200"):
                    ui.icon("check_circle", color="green", size="md")
                    ui.label("Accuracy").classes("text-sm font-semibold text-gray-600 mt-2")
                    ui.label(f"{accuracy:.1f}%").classes("text-3xl font-bold mt-1 text-green-700")

                with ui.column().classes("flex-1 min-w-[220px] rounded-xl p-5 text-center border items-center justify-center shadow-sm bg-blue-50 border-blue-200"):
                    ui.icon("image", color="blue", size="md")
                    ui.label("Total Evaluated").classes("text-sm font-semibold text-gray-600 mt-2")
                    ui.label(f"{len(st.images)}").classes("text-3xl font-bold mt-1 text-blue-700")

                with ui.column().classes("flex-1 min-w-[220px] rounded-xl p-5 text-center border items-center justify-center shadow-sm bg-purple-50 border-purple-200"):
                    ui.icon("memory", color="purple", size="md")
                    ui.label("Memory Footprint").classes("text-sm font-semibold text-gray-600 mt-2")
                    ui.label(f"{s['avg_mem']:.2f} MB").classes("text-3xl font-bold mt-1 text-purple-700")

                with ui.column().classes("flex-1 min-w-[220px] rounded-xl p-5 text-center border items-center justify-center shadow-sm bg-gray-50 border-gray-200"):
                    ui.icon("developer_board", color="gray", size="md")
                    ui.label("Hardware Used").classes("text-sm font-semibold text-gray-600 mt-2")
                    ui.label(f"{get_device().upper()}").classes("text-3xl font-bold mt-1 text-gray-700")

            with ui.row().classes("w-full gap-6 flex-wrap justify-center mt-6"):
                with ui.column().classes("flex-1 min-w-[220px] rounded-xl p-5 text-center border items-center justify-center shadow-sm bg-orange-50 border-orange-200"):
                    ui.icon("timer", color="orange", size="md")
                    ui.label("Avg Inference Time").classes("text-sm font-semibold text-gray-600 mt-2")
                    ui.label(f"{s['avg_time']:.4f}s").classes("text-3xl font-bold mt-1 text-orange-700")

                with ui.column().classes("flex-1 min-w-[220px] rounded-xl p-5 text-center border items-center justify-center shadow-sm bg-indigo-50 border-indigo-200"):
                    ui.icon("analytics", color="indigo", size="md")
                    ui.label("Avg Confidence").classes("text-sm font-semibold text-gray-600 mt-2")
                    ui.label(f"{s['avg_conf']/100:.3f}").classes("text-3xl font-bold mt-1 text-indigo-700")

                with ui.column().classes("flex-1 min-w-[220px] rounded-xl p-5 text-center border items-center justify-center shadow-sm bg-yellow-50 border-yellow-200"):
                    ui.icon("bolt", color="orange", size="md")
                    ui.label("Avg Power").classes("text-sm font-semibold text-gray-600 mt-2")
                    ui.label("N/A").classes("text-3xl font-bold mt-1 text-orange-700")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    global app_state

    print("Pre-loading models (this may take a while)...")
    try:
        load_models()
    except (RuntimeError, OSError, ValueError) as e:
        print(f"Warning: Model loading error: {e}")

    image_dir = Path(__file__).parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    all_images = sorted([f for f in image_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    app_state = AppState(all_images)

    @ui.page("/")
    def index():
        with ui.column().classes("w-full max-w-6xl mx-auto p-6 items-stretch"):
            ui.label("Zero-Shot Object Detection Benchmark").classes("text-4xl font-extrabold mb-8 text-center text-indigo-700")

            if not app_state.images:
                ui.label("No images found in images/ directory. Please add some and refresh.").classes("text-red-500 text-xl text-center")
                return

            if app_state.phase == 1:
                render_phase1()
            elif app_state.phase == 1.5:
                render_phase1_5()
            elif app_state.phase == 2:
                render_phase2()
            elif app_state.phase == 3:
                render_phase3()

    ui.run(title="Zero-Shot Benchmarking", port=8081)


if __name__ in {"__main__", "__mp_main__"}:
    main()
