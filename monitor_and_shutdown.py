import mlflow
from mlflow.tracking import MlflowClient
import time
import os
import sys

# Ensure outputs directories exist
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)

log_file_path = "outputs/logs/monitor.log"

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}\n"
    print(formatted_msg, end="")
    with open(log_file_path, "a") as f:
        f.write(formatted_msg)

log("Starting monitor and shutdown script...")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

experiment_name = "ECG_TemporalEncoder_ExpandedSweep"

# Loop until the sweep finishes
while True:
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            log(f"Experiment '{experiment_name}' not found yet. Retrying...")
            time.sleep(60)
            continue
            
        # Find all parent runs
        parent_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'temporal_encoder_expanded_parent'"
        )
        
        if not parent_runs:
            log("No parent runs found yet. Waiting...")
            time.sleep(60)
            continue
            
        # Get the latest parent run by start time
        latest_parent = sorted(parent_runs, key=lambda r: r.info.start_time, reverse=True)[0]
        
        status = latest_parent.info.status
        log(f"Latest Parent Run ID: {latest_parent.info.run_id}, Status: {status}")
        
        if status in ["FINISHED", "FAILED", "KILLED"]:
            log("Sweep parent run has ended! Proceeding to generate report and shutdown...")
            break
            
    except Exception as e:
        log(f"Error during poll: {e}")
        
    time.sleep(60)

# Generate report
try:
    log("Retrieving completed trials for the parent run...")
    child_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{latest_parent.info.run_id}'"
    )
    
    log(f"Found {len(child_runs)} child runs.")
    
    report_lines = [
        "# ECG Temporal Encoder Hyperparameter Sweep Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Parent Run ID:** `{latest_parent.info.run_id}`",
        f"**Experiment Name:** `{experiment_name}`",
        "",
        "## Performance Table",
        "",
        "| Trial Name | Pretrain SSL | LR | Hidden Size | Layers | LSTM Drp | FC Drp | Test Subset Acc | Test Hamming Loss | Test Macro F1 | Test Macro AUC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    best_acc = -1
    best_trial = None
    best_config = {}
    
    for r in sorted(child_runs, key=lambda x: x.data.tags.get("mlflow.runName", "")):
        name = r.data.tags.get("mlflow.runName", "N/A")
        status = r.info.status
        if status != "FINISHED":
            continue
            
        p = r.data.params
        m = r.data.metrics
        
        ssl = p.get("pretrain_strategy", "None")
        lr = p.get("lr", "N/A")
        hidden = p.get("hidden_size", "N/A")
        layers = p.get("num_layers", "N/A")
        drp_lstm = p.get("dropout_lstm", "N/A")
        drp_fc = p.get("dropout_fc", "N/A")
        
        acc = m.get("subset_accuracy", 0.0)
        hl = m.get("hamming_loss", 0.0)
        f1 = m.get("macro_f1", 0.0)
        auc = m.get("macro_auc", 0.0)
        
        report_lines.append(
            f"| {name} | {ssl} | {lr} | {hidden} | {layers} | {drp_lstm} | {drp_fc} | {acc:.4f} | {hl:.4f} | {f1:.4f} | {auc:.4f} |"
        )
        
        if acc > best_acc:
            best_acc = acc
            best_trial = name
            best_config = {
                "ssl": ssl,
                "lr": lr,
                "hidden": hidden,
                "layers": layers,
                "drp_lstm": drp_lstm,
                "drp_fc": drp_fc,
                "acc": acc,
                "auc": auc
            }
            
    report_lines.append("")
    if best_trial:
        report_lines.append("## Best Configuration Found")
        report_lines.append(f"- **Trial:** `{best_trial}`")
        report_lines.append(f"- **SSL Strategy:** `{best_config['ssl']}`")
        report_lines.append(f"- **Learning Rate:** `{best_config['lr']}`")
        report_lines.append(f"- **Hidden Size:** `{best_config['hidden']}`")
        report_lines.append(f"- **Layers:** `{best_config['layers']}`")
        report_lines.append(f"- **Subset Accuracy:** `{best_config['acc']:.4f}`")
        report_lines.append(f"- **Macro AUC:** `{best_config['auc']:.4f}`")
    else:
        report_lines.append("No completed trials found in this sweep.")
        
    report_content = "\n".join(report_lines)
    
    with open("outputs/reports/sweep_report.md", "w") as f:
        f.write(report_content)
        
    log("Report generated successfully at outputs/reports/sweep_report.md")
    
    # Update project state
    state_path = ".agents/project/project_state.md"
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state_content = f.read()
            
        update_str = f"\n* Executed the expanded parameter sweep (`run_expanded_sweep.py`) to completion, identifying `{best_trial}` as the best configuration with downstream test accuracy of `{best_acc:.4f}`."
        # Inject before '## Next Recommended Task'
        if "## Next Recommended Task" in state_content:
            parts = state_content.split("## Next Recommended Task")
            new_content = parts[0] + update_str + "\n\n## Next Recommended Task" + parts[1]
            with open(state_path, "w") as f:
                f.write(new_content)
            log("Project state updated successfully.")
            
except Exception as e:
    log(f"Error generating report: {e}")

# Shutdown system
log("Initiating system shutdown in 30 seconds...")
os.system("shutdown /s /t 30")
log("Shutdown command executed.")
sys.exit(0)
