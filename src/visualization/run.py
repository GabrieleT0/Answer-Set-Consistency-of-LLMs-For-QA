import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualization.p_value_action_heatmap import main as p_value_action_heatmap_main
from visualization.p_value_pairwise_heatmap import main as p_value_pairwise_heatmap_main
from visualization.bubble_scatter import main as bubble_scatter_main
from visualization.line_chart import main as line_chart_main
from visualization.line_chart_pos_neg import main as line_chart_pos_neg_main
from visualization.io_utils import default_charts_dir, default_results_dir


if __name__ == "__main__":
    config = {
        "folder": default_results_dir(),
        "out_dir": default_charts_dir(),
        "time": None,
        "actions": ["zero-shot", "fixing", "classification", "chain-of-thought"],
    }
    p_value_action_heatmap_main(config)
    p_value_pairwise_heatmap_main(config)
    line_chart_main(config)
    line_chart_pos_neg_main(config)
    bubble_scatter_main(config)
    
