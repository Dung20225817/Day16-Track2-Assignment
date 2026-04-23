variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "hf_token" {
  description = "Hugging Face Token for gated models (like Gemma)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "model_id" {
  description = "Hugging Face Model ID to serve"
  type        = string
  default     = "google/gemma-4-E2B-it"
}

variable "machine_type" {
  description = "GCE Machine Type. GPU: n1-standard-4 | CPU fallback: n2-standard-8"
  type        = string
  default     = "n2-standard-8"  # CPU fallback (no GPU quota needed)
}

variable "gpu_type" {
  description = "GPU accelerator type (ignored when gpu_count = 0)"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "Number of GPUs to attach. Set to 0 for CPU-only mode."
  type        = number
  default     = 0  # 0 = CPU mode (no quota required)
}
