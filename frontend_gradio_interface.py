import gradio as gr
import asyncio
from typing import Tuple
from agents.orchestrator import OrchestratorAgent
from utils.logger import setup_logger

logger = setup_logger(__name__)

class GradioInterface:
    def __init__(self):
        self.orchestrator = OrchestratorAgent()
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        with gr.Blocks(
            title="Autonomous AI Agent System",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .panel {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 20px;
                margin: 10px 0;
            }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🤖 Autonomous AI Agent System
            
            This system can:
            - **Scrape URLs** and store content in a vector database
            - **Answer queries** using scraped content or web search
            - **Automatically route** between local knowledge and web search
            """)
            
            with gr.Tabs():
                # URL Scraping Tab
                with gr.TabItem("🌐 URL Scraping"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            url_input = gr.Textbox(
                                label="URL to Scrape",
                                placeholder="https://example.com",
                                lines=1
                            )
                            deep_scrape_check = gr.Checkbox(
                                label="Deep Scrape (More comprehensive)",
                                value=False
                            )
                            scrape_btn = gr.Button("🔍 Scrape URL", variant="primary")
                        
                        with gr.Column(scale=3):
                            scrape_output = gr.Textbox(
                                label="Scraping Results",
                                lines=10,
                                interactive=False
                            )
                
                # Query Tab
                with gr.TabItem("💬 Ask Questions"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            query_input = gr.Textbox(
                                label="Your Question",
                                placeholder="Ask anything about the scraped content...",
                                lines=3
                            )
                            query_btn = gr.Button("🔍 Ask Question", variant="primary")
                            
                            gr.Markdown("### 📊 Response Source")
                            source_indicator = gr.Textbox(
                                label="Information Source",
                                interactive=False,
                                lines=1
                            )
                        
                        with gr.Column(scale=3):
                            query_output = gr.Textbox(
                                label="Answer",
                                lines=12,
                                interactive=False
                            )
                
                # System Status Tab
                with gr.TabItem("📊 System Status"):
                    with gr.Column():
                        gr.Markdown("### System Components Status")
                        
                        with gr.Row():
                            with gr.Column():
                                mcp_status = gr.Textbox(
                                    label="MCP Tools Status",
                                    value="🟡 Checking...",
                                    interactive=False
                                )
                                pinecone_status = gr.Textbox(
                                    label="Pinecone Vector DB Status",
                                    value="🟡 Checking...",
                                    interactive=False
                                )
                            
                            with gr.Column():
                                tavily_status = gr.Textbox(
                                    label="Tavily Search Status",
                                    value="🟡 Checking...",
                                    interactive=False
                                )
                                embedding_status = gr.Textbox(
                                    label="Embedding Service Status",
                                    value="🟡 Checking...",
                                    interactive=False
                                )
                        
                        status_refresh_btn = gr.Button("🔄 Refresh Status")
                        
                        gr.Markdown("### 📝 Recent Activity Log")
                        activity_log = gr.Textbox(
                            label="Activity Log",
                            lines=8,
                            interactive=False,
                            value="System initialized..."
                        )
            
            # Event handlers
            scrape_btn.click(
                fn=self.handle_scraping,
                inputs=[url_input, deep_scrape_check],
                outputs=[scrape_output]
            )
            
            query_btn.click(
                fn=self.handle_query,
                inputs=[query_input],
                outputs=[query_output, source_indicator]
            )
            
            status_refresh_btn.click(
                fn=self.check_system_status,
                outputs=[mcp_status, pinecone_status, tavily_status, embedding_status]
            )
            
            # Load initial status
            interface.load(
                fn=self.check_system_status,
                outputs=[mcp_status, pinecone_status, tavily_status, embedding_status]
            )
        
        return interface
    
    def handle_scraping(self, url: str, deep_scrape: bool) -> str:
        """Handle URL scraping request"""
        if not url:
            return "❌ Please provide a valid URL"
        
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.orchestrator.process_url_scraping(url, deep_scrape)
            )
            loop.close()
            
            if result.get("success"):
                return f"""✅ Successfully scraped and stored content!

📊 **Results:**
- Chunks stored: {result.get('chunks_stored', 0)}
- Source URL: {result.get('url', '')}

📝 **Content Preview:**
{result.get('content_preview', 'No preview available')}

The content has been processed and stored in the vector database for future queries."""
            else:
                return f"❌ Scraping failed: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"Error in scraping handler: {str(e)}")
            return f"❌ Error: {str(e)}"
    
    def handle_query(self, query: str) -> Tuple[str, str]:
        """Handle query request"""
        if not query:
            return "❌ Please provide a question", "No source"
        
        try:
            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.orchestrator.process_query(query)
            )
            loop.close()
            
            if result.get("success"):
                answer = result.get("answer", "No answer provided")
                source = result.get("source", "unknown")
                
                # Format source indicator
                if source == "vector_database":
                    source_text = f"📚 Vector Database ({result.get('num_sources', 0)} sources)"
                elif source == "web_search":
                    source_text = "🌐 Web Search (Tavily)"
                else:
                    source_text = f"❓ {source}"
                
                return answer, source_text
            else:
                error_msg = f"❌ Query failed: {result.get('error', 'Unknown error')}"
                return error_msg, "Error"
                
        except Exception as e:
            logger.error(f"Error in query handler: {str(e)}")
            return f"❌ Error: {str(e)}", "Error"
    
    def check_system_status(self) -> Tuple[str, str, str, str]:
        """Check status of all system components"""
        try:
            # Check MCP Tools
            try:
                from services.mcp_client import MCPClient
                mcp_client = MCPClient()
                mcp_status = "🟢 Connected"
            except Exception:
                mcp_status = "🔴 Connection Failed"
            
            # Check Pinecone
            try:
                from services.pinecone_service import PineconeService
                pinecone_service = PineconeService()
                pinecone_status = "🟢 Connected"
            except Exception:
                pinecone_status = "🔴 Connection Failed"
            
            # Check Tavily
            try:
                from services.tavily_service import TavilyService
                tavily_service = TavilyService()
                tavily_status = "🟢 Connected"
            except Exception:
                tavily_status = "🔴 Connection Failed"
            
            # Check Embedding Service
            try:
                from services.embedding_service import EmbeddingService
                embedding_service = EmbeddingService()
                embedding_status = "🟢 Model Loaded"
            except Exception:
                embedding_status = "🔴 Model Load Failed"
            
            return mcp_status, pinecone_status, tavily_status, embedding_status
            
        except Exception as e:
            logger.error(f"Error checking system status: {str(e)}")
            error_status = f"🔴 Error: {str(e)}"
            return error_status, error_status, error_status, error_status