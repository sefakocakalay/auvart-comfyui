"""
AUVART BATCH TESTER - ComfyUI API Parameter Sweep
===================================================
Systematically tests different ControlNet strength/end/CFG combinations
via ComfyUI's API and saves all results for comparison.

Usage:
  python batch_tester.py --workflow path/to/workflow.json --output-dir ./results/
  python batch_tester.py --workflow wf.json --output-dir ./results/ --strengths 0.5,0.7,0.9,1.1,1.3
  python batch_tester.py --workflow wf.json --output-dir ./results/ --cfgs 5,6,7,8 --ends 0.6,0.7,0.8,0.9

Requirements: pip install requests Pillow
"""

import argparse
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from itertools import product


COMFYUI_URL = "http://127.0.0.1:8188"


def queue_prompt(workflow, client_id="auvart_batch"):
    """Queue a workflow via ComfyUI API."""
    data = json.dumps({
        "prompt": workflow,
        "client_id": client_id
    }).encode('utf-8')

    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    response = urllib.request.urlopen(req)
    return json.loads(response.read())


def get_history(prompt_id):
    """Get generation history for a prompt."""
    req = urllib.request.Request(f"{COMFYUI_URL}/history/{prompt_id}")
    response = urllib.request.urlopen(req)
    return json.loads(response.read())


def download_image(filename, subfolder, folder_type, save_path):
    """Download a generated image from ComfyUI."""
    params = urllib.parse.urlencode({
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    })
    req = urllib.request.Request(f"{COMFYUI_URL}/view?{params}")
    response = urllib.request.urlopen(req)

    with open(save_path, 'wb') as f:
        f.write(response.read())


def wait_for_completion(prompt_id, timeout=300, poll_interval=2):
    """Wait for a prompt to complete."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            history = get_history(prompt_id)
            if prompt_id in history:
                return history[prompt_id]
        except Exception:
            pass
        time.sleep(poll_interval)
    raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout}s")


def find_nodes(workflow, node_type):
    """Find all nodes of a given type in a workflow."""
    results = []
    for node_id, node in workflow.items():
        if node.get("class_type") == node_type:
            results.append((node_id, node))
    return results


def find_preview_nodes(workflow):
    """Find PreviewImage or SaveImage nodes."""
    results = []
    for node_id, node in workflow.items():
        if node.get("class_type") in ("PreviewImage", "SaveImage"):
            results.append(node_id)
    return results


def load_workflow_api_format(workflow_path):
    """
    Load a workflow JSON. If it's in the UI format (has 'nodes' array),
    convert to API format. If already API format, return as-is.
    """
    with open(workflow_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if this is UI format (has 'nodes' as array) or API format (dict of node_id: node)
    if isinstance(data, dict) and 'nodes' in data and isinstance(data['nodes'], list):
        print("NOTE: Workflow is in UI format. For batch testing, export")
        print("      the workflow as API format from ComfyUI:")
        print("      Menu > Save (API Format)")
        print("")
        print("      Alternatively, load the workflow in ComfyUI first,")
        print("      then use the /api endpoint to get the API format.")
        raise ValueError(
            "Workflow is in UI format. Export as API format from ComfyUI "
            "(Menu > Enable Dev Mode > Save API Format)"
        )

    return data


def run_parameter_sweep(workflow_path, output_dir, strengths=None,
                        ends=None, cfgs=None, seed=None):
    """
    Run a parameter sweep varying CN strength, end%, and CFG.

    Args:
        workflow_path: Path to ComfyUI workflow JSON (API format)
        output_dir: Directory to save results
        strengths: List of CN strength values to test
        ends: List of CN end% values to test
        cfgs: List of CFG values to test
        seed: Fixed seed (None = random each time)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow = load_workflow_api_format(workflow_path)

    if strengths is None:
        strengths = [0.7, 0.85, 1.0, 1.2, 1.5]
    if ends is None:
        ends = [0.65, 0.80, 1.0]
    if cfgs is None:
        cfgs = [7]

    # Find ControlNet and KSampler nodes
    cn_nodes = find_nodes(workflow, "ControlNetApplyAdvanced")
    ks_nodes = find_nodes(workflow, "KSampler")
    preview_nodes = find_preview_nodes(workflow)

    if not cn_nodes:
        print("ERROR: No ControlNetApplyAdvanced node found in workflow")
        return
    if not ks_nodes:
        print("ERROR: No KSampler node found in workflow")
        return

    cn_id = cn_nodes[0][0]  # First CN node
    ks_id = ks_nodes[0][0]  # First KSampler (Pass 1)

    combinations = list(product(strengths, ends, cfgs))
    total = len(combinations)

    print(f"=== AUVART BATCH TESTER ===")
    print(f"Workflow: {workflow_path}")
    print(f"Output: {output_dir}")
    print(f"CN node: {cn_id}")
    print(f"KSampler node: {ks_id}")
    print(f"Strengths: {strengths}")
    print(f"End values: {ends}")
    print(f"CFG values: {cfgs}")
    print(f"Total combinations: {total}")
    print(f"{'=' * 50}")

    results = []
    for i, (strength, end, cfg) in enumerate(combinations, 1):
        print(f"\n[{i}/{total}] strength={strength}, end={end}, cfg={cfg}")

        # Modify workflow
        wf = json.loads(json.dumps(workflow))  # deep copy

        # Set CN strength and end
        if "inputs" in wf[cn_id]:
            wf[cn_id]["inputs"]["strength"] = strength
            wf[cn_id]["inputs"]["end_percent"] = end

        # Set CFG
        if "inputs" in wf[ks_id]:
            wf[ks_id]["inputs"]["cfg"] = cfg
            if seed is not None:
                wf[ks_id]["inputs"]["seed"] = seed

        # Queue
        try:
            result = queue_prompt(wf)
            prompt_id = result["prompt_id"]
            print(f"  Queued: {prompt_id}")

            # Wait for completion
            history = wait_for_completion(prompt_id)
            print(f"  Completed!")

            # Download results
            outputs = history.get("outputs", {})
            for node_id in preview_nodes:
                if node_id in outputs:
                    images = outputs[node_id].get("images", [])
                    for img_info in images:
                        filename = f"str{strength}_end{end}_cfg{cfg}.png"
                        save_path = output_dir / filename
                        download_image(
                            img_info["filename"],
                            img_info.get("subfolder", ""),
                            img_info.get("type", "temp"),
                            save_path
                        )
                        print(f"  Saved: {save_path}")
                        results.append({
                            "strength": strength,
                            "end": end,
                            "cfg": cfg,
                            "file": str(save_path)
                        })
                        break  # Only save first preview
                    break  # Only from first preview node

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "strength": strength,
                "end": end,
                "cfg": cfg,
                "error": str(e)
            })

    # Save results log
    log_path = output_dir / "results.json"
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'=' * 50}")
    print(f"Done! {len([r for r in results if 'file' in r])}/{total} successful")
    print(f"Results log: {log_path}")

    # Generate HTML comparison
    generate_html_comparison(results, output_dir)


def generate_html_comparison(results, output_dir):
    """Generate an HTML page for visual comparison of results."""
    html_path = output_dir / "comparison.html"

    successful = [r for r in results if 'file' in r]
    if not successful:
        return

    rows = []
    for r in successful:
        filename = Path(r['file']).name
        rows.append(f"""
        <div class="card">
            <img src="{filename}" loading="lazy">
            <div class="label">
                str: {r['strength']} | end: {r['end']} | cfg: {r['cfg']}
            </div>
        </div>""")

    html = f"""<!DOCTYPE html>
<html><head>
<title>Auvart Batch Test Results</title>
<style>
body {{ background: #1a1a1a; color: #fff; font-family: monospace; padding: 20px; }}
h1 {{ text-align: center; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }}
.card {{ background: #2a2a2a; border-radius: 8px; overflow: hidden; width: 300px; }}
.card img {{ width: 100%; display: block; }}
.label {{ padding: 8px; text-align: center; font-size: 12px; }}
</style>
</head><body>
<h1>Auvart Batch Test Results</h1>
<p style="text-align:center">{len(successful)} variants generated</p>
<div class="grid">
{''.join(rows)}
</div>
</body></html>"""

    with open(html_path, 'w') as f:
        f.write(html)
    print(f"HTML comparison: {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Auvart Batch Tester - ComfyUI Parameter Sweep')
    parser.add_argument('--workflow', required=True,
                        help='Path to workflow JSON (API format)')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for results')
    parser.add_argument('--strengths', type=str,
                        default='0.7,0.85,1.0,1.2,1.5',
                        help='CN strengths (comma-separated)')
    parser.add_argument('--ends', type=str,
                        default='0.65,0.80,1.0',
                        help='CN end values (comma-separated)')
    parser.add_argument('--cfgs', type=str, default='7',
                        help='CFG values (comma-separated)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Fixed seed (default: random)')
    parser.add_argument('--url', type=str, default='http://127.0.0.1:8188',
                        help='ComfyUI API URL')

    args = parser.parse_args()

    global COMFYUI_URL
    COMFYUI_URL = args.url

    strengths = [float(x) for x in args.strengths.split(',')]
    ends = [float(x) for x in args.ends.split(',')]
    cfgs = [float(x) for x in args.cfgs.split(',')]

    run_parameter_sweep(
        args.workflow, args.output_dir,
        strengths=strengths, ends=ends, cfgs=cfgs,
        seed=args.seed
    )


if __name__ == '__main__':
    main()