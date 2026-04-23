output "load_balancer_ip" {
  description = "External IP of the Load Balancer"
  value       = google_compute_global_forwarding_rule.vllm_fwd.ip_address
}

output "health_check_url" {
  description = "Health check endpoint — should return {status: ok}"
  value       = "http://${google_compute_global_forwarding_rule.vllm_fwd.ip_address}/health"
}

output "benchmark_result_url" {
  description = "View benchmark results (after running benchmark.py)"
  value       = "http://${google_compute_global_forwarding_rule.vllm_fwd.ip_address}/v1/benchmark/result"
}

output "benchmark_run_url" {
  description = "Trigger benchmark via HTTP (alternative to SSH)"
  value       = "http://${google_compute_global_forwarding_rule.vllm_fwd.ip_address}/v1/benchmark/run"
}

output "vm_name" {
  description = "Name of the Compute Engine VM instance"
  value       = google_compute_instance.gpu_node.name
}

output "iap_ssh_command" {
  description = "SSH into the VM via IAP (no external IP needed)"
  value       = "gcloud compute ssh ${google_compute_instance.gpu_node.name} --zone=${google_compute_instance.gpu_node.zone} --tunnel-through-iap --project=${var.project_id}"
}

output "run_benchmark_command" {
  description = "Command to run benchmark after SSH-ing in"
  value       = "python3 /opt/ml-benchmark/benchmark.py"
}

output "view_logs_command" {
  description = "Watch startup progress"
  value       = "gcloud compute ssh ${google_compute_instance.gpu_node.name} --zone=${google_compute_instance.gpu_node.zone} --tunnel-through-iap --project=${var.project_id} --command='sudo tail -f /var/log/startup.log'"
}
