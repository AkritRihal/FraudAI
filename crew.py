import gradio as gr
from tasks import fraud_task, score_task, summary_task
from agents import run_agent

def process_message(user_input):
    # Run each task using the run_agent function
    fraud_response = run_agent(fraud_task.description.format(topic=user_input))
    score_response = run_agent(score_task.description.format(topic=user_input))
    summary_response = run_agent(summary_task.description.format(topic=user_input))

    # Aggregate the responses
    output = {
        "Fraud Classification": fraud_response,
        "Reliability Score": score_response,
        "Summary": summary_response
    }
    return output["Fraud Classification"], output["Reliability Score"], output["Summary"]

def main():
    # Gradio interface setup
    with gr.Blocks(css="""
        .gradio-container {
            background-color: #e6f7ff; 
            padding: 30px; 
            border-radius: 15px; 
            font-family: 'Arial', sans-serif;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        #title {
            color: #007bff; 
            font-size: 2.5em; 
            text-align: center;
        }
        #description {
            color: #555; 
            font-size: 1.2em; 
            text-align: center;
            margin-bottom: 20px;
        }
        .output-label {
            font-weight: bold; 
            font-size: 1em; 
            color: #333;
        }
        /* Slider styles */
        .slider {
            margin: 10px 0;
            background: linear-gradient(to right, red, yellow, green); 
            border-radius: 5px;
            height: 10px; 
        }
        .slider .slider-thumb {
            background-color: white; 
            border-radius: 50%; 
            border: 2px solid #007bff; 
        }
        #submit-button {
            background-color: #007bff; 
            color: white; 
            border-radius: 5px; 
            padding: 10px 20px;
            border: none;
        }
        #submit-button:hover {
            background-color: #0056b3;
        }
    """) as demo:
        gr.Markdown("<h1 id='title'>Message Lens</h1>", elem_id="title")
        gr.Markdown("<p id='description'>This application analyzes your message for fraud classification, reliability scoring, and provides a summary.</p>", elem_id="description")
        
        # Input window for user message
        user_input = gr.Textbox(label="Enter your message:", placeholder="Type your message here...", lines=2, elem_id="user-input")
        
        # Output areas
        fraud_label = gr.Label(label="Fraud Classification", elem_id="fraud-label")
        reliability_slider = gr.Slider(label="Reliability Score", minimum=0, maximum=10, elem_id="reliability-slider", interactive=False)  
        summary_label = gr.Label(label="Summary", elem_id="summary-label")
        
        # Button to process the input
        submit_button = gr.Button("Analyze", elem_id="submit-button", variant="primary")
        
        # Set up the button to call the processing function
        submit_button.click(fn=process_message, inputs=user_input, outputs=[fraud_label, reliability_slider, summary_label])

    # Launch the Gradio app
    demo.launch()

if __name__ == "__main__":
    main()