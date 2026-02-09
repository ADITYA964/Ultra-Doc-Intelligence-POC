import gradio as gr
import requests
import json, re
from tenacity import retry, stop_after_attempt, wait_exponential

API_BASE = "http://localhost:8000"

class GradioDocIntelligence:
    def __init__(self):
        self.doc_info = None
    
    # Retry upload if network flutters, but stop if server returns 500 (Poppler error)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6))
    def _make_upload_request(self, file_path):
        with open(file_path, 'rb') as f:
            files = {"file": (file_path.split('/')[-1], f, "application/pdf")}
            return requests.post(f"{API_BASE}/upload", files=files, timeout=60)

    def upload_document(self, file):
        if file is None:
            return "Please upload a file", None, None, gr.update(visible=False)
        
        try:
            # Using the retry-enabled helper
            response = self._make_upload_request(file.name)
            
            if response.status_code == 200:
                self.doc_info = response.json()
                status = f"✅ Processed: {self.doc_info['filename']}\nID: {self.doc_info['document_id']}"
                return status, self.doc_info['document_id'], self.doc_info['filename'], gr.update(visible=True)
            
            # Specific hint for the Poppler error you saw
            error_msg = response.text
            if "PDFInfoNotInstalledError" in error_msg or "poppler" in error_msg.lower():
                return "❌ Server Error: Poppler is not installed on the API server.", None, None, gr.update(visible=False)
                
            return f"❌ Upload failed: {error_msg}", None, None, gr.update(visible=False)
            
        except Exception as e:
            return f"❌ Connection Error: {str(e)}", None, None, gr.update(visible=False)
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=4))
    def ask_question(self, question, doc_id):
        if not question.strip() or not doc_id:
            return "Ask a question after uploading", "", 0.0, "", False
        
        try:
            response = requests.post(f"{API_BASE}/ask", json={
                "question": question, "document_id": doc_id
            }, timeout=2000)
            
            if response.status_code == 200:
                result = response.json()
                sources_html = ""
                # Added protection in case 'sources' is empty
                # Define logistics keywords to highlight within chunks
                keywords = ["Reference ID", "Agreed Amount", "Load ID", "Weight", "Commodity", "Pickup", "Drop"]
                for source in result.get('sources', []):
                    
                    content = source.get('content', '')
                    
                    # Apply Regex highlighting for specific logistics terms
                    for kw in keywords:
                        # This regex finds the keyword and the text immediately following it until a newline or significant gap
                        # content = re.sub(f"({kw}.*?)(?=\\n|$)", r"<mark style='background: #ffeb3b; padding: 2px; font-weight: bold;'>\1</mark>", content, flags=re.IGNORECASE)
                        content = re.sub(f"({re.escape(kw)}.*?)(?=\\n|[:;.,!?]|$)", r"<mark style='background: #4fc3f7; color: white; padding: 2px 4px; border-radius: 3px; font-weight: bold;'>\1</mark>", content, flags=re.IGNORECASE)
                        
                    sources_html += f"""
                    <div style="margin: 10px 0; padding: 12px; background-color: #f0f7ff; border-left: 5px solid #007bff; border-radius: 4px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <strong style="color: #0056b3;">Chunk {source.get('id', 'N/A')+1}</strong>
                            <span style="background: #007bff; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;">
                                Similarity: {source.get('score', 0):.2f}
                            </span>
                        </div>
                        <div style="color: #333; font-family: 'Courier New', monospace; font-size: 0.95em; line-height: 1.5; background: #ffffff; padding: 10px; border: 1px solid #d1d9e0; border-radius: 4px;">
                            {content}
                        </div>
                    </div>
                    """
                
                # Update reliability status text
                # rel_status = "✅ Reliable" if result.get('is_reliable') else "⚠️ Low Confidence"
                is_reliable = result.get('is_reliable')
                rel_color = "#28a745" if is_reliable else "#ffc107"
                rel_text = "✅ Reliable" if is_reliable else "⚠️ Low Confidence"
                
                rel_status = f"<span style='color: {rel_color}; font-weight: bold;'>{rel_text}</span>"
                
                return result['answer'], sources_html, result['confidence_score'], result['confidence_reason'], rel_status
            return "❌ API Error", "", 0.0, "", "Status: Error"
        except Exception as e:
            return f"❌ Error: {str(e)}", "", 0.0, "", "Status: Error"

    def extract_structured(self, doc_id):
        if not doc_id:
            return "Upload document first", gr.update(visible=False)
        try:
            response = requests.post(f"{API_BASE}/extract", json={"document_id": doc_id}, timeout=2000)
            if response.status_code == 200:
                # Returns formatted JSON string for display
                return f"```json\n{json.dumps(response.json(), indent=2)}\n```", gr.update(visible=True)
            return f"❌ Error: {response.text}", gr.update(visible=False)
        except Exception as e:
            return f"❌ Error: {str(e)}", gr.update(visible=False)

app = GradioDocIntelligence()

with gr.Blocks(title="Ultra Doc-Intelligence", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚚 Ultra Doc-Intelligence")
    gr.Markdown("Upload logistics docs & ask questions with sources + confidence scores")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📤 Upload Document")
            file_input = gr.File(label="PDF/DOCX/TXT", file_types=[".pdf", ".docx", ".txt"])
            upload_btn = gr.Button("📤 Process", variant="primary")
            doc_status = gr.Markdown("")
            # Hidden textboxes to store state
            doc_id_state = gr.Textbox(visible=False)
            filename_state = gr.Textbox(visible=False)
        
        with gr.Column(scale=2):
            gr.Markdown("## 📦 Extract Shipment Data")
            extract_btn = gr.Button("🔍 Extract JSON", variant="secondary")
            structured_json = gr.Markdown("Click Extract to see structured data...", visible=False)
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("## 💬 Ask Questions")
            question_input = gr.Textbox(label="e.g. What is the carrier rate?", lines=2)
            ask_btn = gr.Button("🤖 Answer", variant="primary")
            answer_output = gr.Markdown("")
            sources_html = gr.HTML(label="Sources")
        with gr.Column(scale=1):
            gr.Markdown("## 📊 Confidence")
            confidence_score = gr.Slider(0, 1, value=0, interactive=False, label="Confidence Score")
            confidence_reason = gr.Textbox(label="Reason", interactive=False)
            reliability_label = gr.Markdown("**Status: --**")
    
    # Event Listeners
    upload_btn.click(
        app.upload_document, 
        inputs=file_input, 
        outputs=[doc_status, doc_id_state, filename_state, structured_json]
    )
    
    ask_btn.click(
        app.ask_question, 
        inputs=[question_input, doc_id_state], 
        outputs=[answer_output, sources_html, confidence_score, confidence_reason, reliability_label]
    )
    
    extract_btn.click(
        app.extract_structured, 
        inputs=doc_id_state, 
        outputs=[structured_json, structured_json]
    )

if __name__ == "__main__":
    demo.launch()
