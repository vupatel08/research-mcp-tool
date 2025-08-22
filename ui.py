"""
Research Tracker MCP Server - Gradio User Interface

Web interface for demonstrating and testing the research inference tools.
"""

import gradio as gr

from mcp_tools import (
    infer_authors, infer_paper_url, infer_code_repository, infer_research_name,
    classify_research_url, infer_publication_date, infer_model, infer_dataset,
    infer_space, infer_license
)
from discovery import find_research_relationships, process_research_relationships, process_research_relationships_live, process_research_relationships_realtime


# Create minimal black/white Gradio interface
with gr.Blocks(title="Research Discovery Engine", theme=gr.themes.Soft(primary_hue="gray", neutral_hue="gray")) as demo:
    gr.Markdown("# Research Discovery Engine")
    gr.Markdown("Enter a research URL and watch the discovery process unfold.")
    
    with gr.Row():
        with gr.Column(scale=4):
            input_text = gr.Textbox(
                label="Research URL",
                placeholder="https://arxiv.org/abs/2506.18787",
                lines=1,
                container=False
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("Discover", variant="primary", size="lg")
    
    # Discovery Log Section
    gr.Markdown("## Discovery Process")
    
    # Progress indicator
    progress_bar = gr.Progress()
    
    log_output = gr.Textbox(
        label="Live Discovery Log",
        lines=12,
        interactive=False,
        container=False,
        value="🚀 Ready to discover...\n\nEnter a research URL above and click 'Discover' to watch the real-time discovery process unfold!\n\nYou'll see:\n• URL analysis\n• Resource discovery steps\n• Cross-referencing progress\n• Final results and metrics"
    )
    
    # Results Section
    gr.Markdown("## Results")
    with gr.Row():
        with gr.Column():
            paper_output = gr.Textbox(label="Paper", interactive=False, container=False)
            code_output = gr.Textbox(label="Code Repository", interactive=False, container=False)
            authors_output = gr.Textbox(label="Authors", interactive=False, container=False)
            
        with gr.Column():
            model_output = gr.Textbox(label="Model", interactive=False, container=False)
            dataset_output = gr.Textbox(label="Dataset", interactive=False, container=False)
            space_output = gr.Textbox(label="Demo Space", interactive=False, container=False)
    
    # Summary Section
    gr.Markdown("## Summary")
    summary_output = gr.Textbox(label="Discovery Summary", interactive=False, container=False)
    
    # Examples Section (moved here after components are defined)
    gr.Examples(
        examples=[
            ["https://arxiv.org/abs/2506.18787"],
            ["https://huggingface.co/papers/2010.11929"],
            ["https://github.com/facebookresearch/segment-anything"],
            ["https://microsoft.github.io/TRELLIS/"]
        ],
        inputs=[input_text],
        outputs=[
            paper_output, code_output, authors_output, 
            model_output, dataset_output, space_output, log_output, summary_output
        ],
        fn=process_research_relationships_realtime,
        cache_examples=False,
        label="Example URLs"
    )
    
    # Connect the interface
    submit_btn.click(
        fn=process_research_relationships_realtime,
        inputs=[input_text],
        outputs=[
            paper_output, code_output, authors_output, 
            model_output, dataset_output, space_output, log_output, summary_output
        ]
    )
    
    # Also trigger on Enter key
    input_text.submit(
        fn=process_research_relationships_realtime,
        inputs=[input_text],
        outputs=[
            paper_output, code_output, authors_output, 
            model_output, dataset_output, space_output, log_output, summary_output
        ]
    )
    
    # Expose MCP tools
    gr.api(infer_authors)
    gr.api(infer_paper_url)
    gr.api(infer_code_repository)
    gr.api(infer_research_name)
    gr.api(classify_research_url)
    gr.api(infer_publication_date)
    gr.api(infer_model)
    gr.api(infer_dataset)
    gr.api(infer_space)
    gr.api(infer_license)
    gr.api(find_research_relationships)
