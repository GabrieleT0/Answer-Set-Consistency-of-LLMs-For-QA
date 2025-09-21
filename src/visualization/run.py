import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualization.p_value_action_heatmap import main as p_value_action_heatmap_main
from visualization.p_value_pairwise_heatmap import main as p_value_pairwise_heatmap_main
from visualization.bubble_scatter import main as bubble_scatter_main
from visualization.line_chart import main as line_chart_main
from visualization.line_chart_pos_neg import main as line_chart_pos_neg_main


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__name__))
    config = {
            "time": "2025-09-22_00-41",
        } 
    p_value_action_heatmap_main(config)
    p_value_pairwise_heatmap_main(config)
    line_chart_main(config)
    line_chart_pos_neg_main(config)
    bubble_scatter_main(config)
    

