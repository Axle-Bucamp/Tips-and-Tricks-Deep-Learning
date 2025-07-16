# 🛠️ AI Tooling: Essential Stack

## ☁️ Cloud-Based Tools (Quick Start, Pay-as-you-go)

| Tool             | Use Case                          |
|------------------|------------------------------------|
| Google Colab     | Notebook prototyping w/ GPU        |
| AWS SageMaker    | Production-grade model ops         |
| Vertex AI (GCP)  | Enterprise ML Ops & pipelines      |
| Hugging Face     | Hosting, Transformers, Spaces      |
| Kaggle Kernels   | Lightweight GPU compute            |

---

# 🔌 Hardware & GPU Drivers

## For Training & Inference:

- **NVIDIA CUDA Toolkit** *(Recommended)*  
  Industry standard for GPU-accelerated training.

- **AMD ROCm** *(Experimental but improving)*  
  Open alternative with growing support.

- **OpenCL / OpenCUDA**  
  Some success on edge devices and less common setups.

---

## Guidelines by Use Case:

- **LLMs:**  
  20–30GB VRAM or advanced offloading techniques required.

- **Vision Models:**
  - **Images:** 12–32GB VRAM
  - **Video:** 48–100GB VRAM for efficient processing pipelines.

---

## Edge/DIY Builds:

- **Raspberry Pi Cluster**  
  CPU-based inference and experimentation.

- **Jetson Orin / Xavier**  
  NVIDIA-powered edge inference solutions.
