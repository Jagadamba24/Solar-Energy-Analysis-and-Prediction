import json
import sys
import io
import base64
import os

# We will execute each code cell in a clean namespace
nb_path = "notebooks/solar_analysis.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

global_scope = {}
execution_count = 1

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        code_str = "".join(cell["source"])
        print(f"Executing cell {execution_count}...")
        
        # Intercept stdout
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        # Clear any old plots
        plt.close("all")
        
        try:
            # Change directory to notebooks folder during execution so relative paths like ../dataset work
            os.chdir("notebooks")
            exec(code_str, global_scope)
        except Exception as e:
            print(f"Error in cell {execution_count}: {e}", file=sys.stderr)
        finally:
            os.chdir("..")
            sys.stdout = old_stdout
            
        stdout_text = redirected_output.getvalue()
        outputs = []
        
        if stdout_text:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in stdout_text.splitlines()]
            })
            
        # Check if any matplotlib figures were generated
        fig_nums = plt.get_fignums()
        for fignum in fig_nums:
            fig = plt.figure(fignum)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            outputs.append({
                "data": {
                    "image/png": img_b64,
                    "text/plain": ["<Figure size ...>"]
                },
                "metadata": {},
                "output_type": "display_data"
            })
            plt.close(fig)
            
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs
        execution_count += 1

with open("notebooks/solar_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

with open("Solar_Energy_Prediction/notebooks/solar_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("\nNotebook executed and all outputs/charts populated successfully!")
