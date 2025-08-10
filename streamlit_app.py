# streamlit_app.py
import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import uuid
import base64
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Multi-Agent AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #1f77b4;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .source-citation {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MultiAgentAPI:
    """API client for the multi-agent system"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def get_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        try:
            response = requests.get(f"{self.base_url}/config", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_sql_tables(self) -> Dict[str, Any]:
        """Get SQL tables information"""
        try:
            response = requests.get(f"{self.base_url}/sql/tables", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self, message: str, user_id: str, conversation_id: Optional[str] = None, 
            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send chat message"""
        try:
            payload = {
                "message": message,
                "user_id": user_id,
                "context": context or {}
            }
            if conversation_id:
                payload["conversation_id"] = conversation_id
            
            response = requests.post(
                f"{self.base_url}/chat", 
                json=payload, 
                timeout=60
            )
            
            # Handle different response types
            if response.status_code == 200:
                try:
                    return response.json()
                except:
                    # If JSON parsing fails, return the text content
                    return {"response": response.text, "conversation_id": "unknown", "agent_used": "unknown"}
            else:
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def get_conversation(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """Get conversation history"""
        try:
            response = requests.get(
                f"{self.base_url}/conversations/{user_id}/{conversation_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def list_conversations(self, user_id: str, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """List user conversations"""
        try:
            response = requests.get(
                f"{self.base_url}/conversations/{user_id}",
                params={"limit": limit, "offset": offset},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def delete_conversation(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """Delete conversation"""
        try:
            response = requests.delete(
                f"{self.base_url}/conversations/{user_id}/{conversation_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

# Initialize API client
@st.cache_resource
def get_api_client():
    return MultiAgentAPI(API_BASE_URL)

api = get_api_client()

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_config" not in st.session_state:
    st.session_state.system_config = None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Multi-Agent AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    setup_sidebar()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chat", 
        "📊 System Status", 
        "🗃️ Database Explorer", 
        "📋 Conversation History", 
        "⚙️ Settings"
    ])
    
    with tab1:
        chat_interface()
    
    with tab2:
        system_status()
    
    with tab3:
        database_explorer()
    
    with tab4:
        conversation_history()
    
    with tab5:
        settings_page()

def setup_sidebar():
    """Setup sidebar with user info and quick actions"""
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # User ID
        col1, col2 = st.columns([3, 1])
        with col1:
            new_user_id = st.text_input("User ID", value=st.session_state.user_id)
        with col2:
            if st.button("🔄", help="Generate new User ID"):
                st.session_state.user_id = str(uuid.uuid4())[:8]
                st.rerun()
        
        if new_user_id != st.session_state.user_id:
            st.session_state.user_id = new_user_id
        
        st.divider()
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🆕 New Conversation", use_container_width=True):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.success("Started new conversation!")
        
        if st.button("🔄 Refresh System Status", use_container_width=True):
            st.cache_data.clear()
            st.success("Status refreshed!")
        
        # System health indicator
        st.divider()
        st.subheader("🏥 System Health")
        
        health = api.health_check()
        if "error" not in health:
            if health.get("status") == "healthy":
                st.success("🟢 System Online")
            else:
                st.warning("🟡 System Issues")
        else:
            st.error("🔴 System Offline")
        
        # Environment info
        if st.session_state.system_config:
            config = st.session_state.system_config
            st.info(f"**Environment:** {config.get('environment', 'Unknown')}")
            
            services = config.get('services', {})
            if services:
                st.write("**Services:**")
                for service, info in services.items():
                    if isinstance(info, dict):
                        st.write(f"• {service}: {info.get('type', 'N/A')}")

def chat_interface():
    """Main chat interface"""
    st.header("💬 Conversational AI Chat")
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    # Check if we have a next message to pre-fill
    initial_message = ""
    if hasattr(st.session_state, 'next_message'):
        initial_message = st.session_state.next_message
        del st.session_state.next_message  # Clear it after use
    
    with col1:
        user_message = st.text_area(
            "Message", 
            value=initial_message,
            placeholder="Ask me about your data, search for information, or explore your database...",
            height=100,
            key="chat_input"
        )
    
    with col2:
        st.write("")  # Spacing
        send_button = st.button("📤 Send", use_container_width=True, type="primary")
        
        # Context options
        with st.expander("🎯 Context Options"):
            department = st.selectbox("Department", ["", "sales", "hr", "finance", "engineering"])
            priority = st.selectbox("Priority", ["", "low", "medium", "high"])
            
            context = {}
            if department:
                context["department"] = department
            if priority:
                context["priority"] = priority
    
    # Send message
    if send_button and user_message.strip():
        with st.spinner("🤔 Thinking..."):
            response = api.chat(
                message=user_message,
                user_id=st.session_state.user_id,
                conversation_id=st.session_state.conversation_id,
                context=context
            )
            
            if "error" not in response:
                # Handle successful response
                conversation_id = response.get("conversation_id", str(uuid.uuid4()))
                agent_used = response.get("agent_used", "unknown")
                response_text = response.get("response", "No response received")
                
                # Update conversation ID
                st.session_state.conversation_id = conversation_id
                
                # Add messages to session state
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.now()
                })
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.now(),
                    "metadata": response
                })
                
                # Show success message
                st.success(f"✅ Response from {agent_used} agent")
                
                # Clear input and refresh
                st.rerun()
                
            else:
                # Handle error response
                error_msg = response.get('error', 'Unknown error occurred')
                st.error(f"❌ Error: {error_msg}")
                
                # Show debug info in expander
                with st.expander("🐛 Debug Information"):
                    st.json(response)
    
    # Display conversation
    if st.session_state.messages:
        st.divider()
        display_conversation()

def display_conversation():
    """Display the conversation messages"""
    for i, message in enumerate(st.session_state.messages):
        is_user = message["role"] == "user"
        
        # Message container
        with st.container():
            if is_user:
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant ({message.get('metadata', {}).get('agent_used', 'unknown')}):</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Display metadata for assistant messages
                metadata = message.get("metadata", {})
                if metadata:
                    display_response_metadata(metadata, i)

def display_response_metadata(metadata: Dict[str, Any], message_index: int):
    """Display response metadata (sources, images, followups)"""
    
    col1, col2, col3 = st.columns(3)
    
    # Sources
    sources = metadata.get("sources", [])
    if sources:
        with col1:
            with st.expander(f"📚 Sources ({len(sources)})"):
                for i, source in enumerate(sources):
                    st.markdown(f"""
                    <div class="source-citation">
                        <strong>{source.get('title', f'Source {i+1}')}</strong><br>
                        <em>{source.get('source', 'Unknown')}</em><br>
                        <small>{source.get('snippet', '')[:100]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Images
    images = metadata.get("images", [])
    if images:
        with col2:
            with st.expander(f"🖼️ Images ({len(images)})"):
                for i, img_data in enumerate(images):
                    try:
                        # Decode base64 image
                        img_bytes = base64.b64decode(img_data)
                        img = Image.open(io.BytesIO(img_bytes))
                        st.image(img, caption=f"Image {i+1}", use_column_width=True)
                    except Exception as e:
                        st.error(f"Could not display image {i+1}: {e}")
    
    # Follow-up suggestions
    followups = metadata.get("followup_suggestions", [])
    if followups:
        with col3:
            with st.expander(f"💡 Suggestions ({len(followups)})"):
                for suggestion in followups:
                    if st.button(f"💬 {suggestion}", key=f"followup_{message_index}_{hash(suggestion)}"):
                        # Instead of directly modifying session_state, store the suggestion
                        # in a different session state variable
                        st.session_state.next_message = suggestion
                        st.rerun()
def system_status():
    """System status and configuration page"""
    st.header("📊 System Status & Configuration")
    
    # Get system info
    health = api.health_check()
    config = api.get_config()
    
    if "error" not in config:
        st.session_state.system_config = config
    
    # Health status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if "error" not in health:
            status = health.get("status", "unknown")
            if status == "healthy":
                st.metric("🏥 System Health", "Healthy", delta="Online")
            else:
                st.metric("🏥 System Health", "Issues", delta="Problems")
        else:
            st.metric("🏥 System Health", "Offline", delta="Error")
    
    with col2:
        if "error" not in config:
            env = config.get("environment", "unknown")
            st.metric("🌍 Environment", env.title())
    
    with col3:
        if "error" not in health:
            services = health.get("services", {})
            active_services = sum(1 for v in services.values() if v)
            st.metric("⚙️ Active Services", f"{active_services}/{len(services)}")
    
    with col4:
        st.metric("👤 Current User", st.session_state.user_id)
    
    # Detailed health information
    if "error" not in health:
        st.subheader("🔍 Detailed Health Status")
        
        services = health.get("services", {})
        config_info = health.get("configuration", {})
        
        # Services status
        service_data = []
        for service, status in services.items():
            service_data.append({
                "Service": service.title(),
                "Status": "✅ Active" if status else "❌ Inactive",
                "Type": "Core Service"
            })
        
        # Configuration status
        for feature, enabled in config_info.items():
            service_data.append({
                "Service": feature.replace("_", " ").title(),
                "Status": "✅ Enabled" if enabled else "❌ Disabled",
                "Type": "Feature Flag"
            })
        
        if service_data:
            df = pd.DataFrame(service_data)
            st.dataframe(df, use_container_width=True)
    
    # Configuration details
    if "error" not in config:
        st.subheader("⚙️ System Configuration")
        
        # Service configuration
        services_config = config.get("services", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Storage & Memory:**")
            memory_config = services_config.get("memory", {})
            st.json(memory_config)
            
            st.write("**Search Service:**")
            search_config = services_config.get("search", {})
            st.json(search_config)
        
        with col2:
            st.write("**Database Service:**")
            sql_config = services_config.get("sql", {})
            st.json(sql_config)
            
            st.write("**Language Model:**")
            llm_config = services_config.get("llm", {})
            st.json(llm_config)
        
        # Features
        features = config.get("features", {})
        if features:
            st.write("**Feature Flags:**")
            feature_df = pd.DataFrame([
                {"Feature": k.replace("_", " ").title(), "Enabled": "✅" if v else "❌"}
                for k, v in features.items()
            ])
            st.dataframe(feature_df, use_container_width=True, hide_index=True)

def database_explorer():
    """Database schema explorer"""
    st.header("🗃️ Database Explorer")
    
    # Get SQL tables info
    with st.spinner("Loading database schema..."):
        tables_info = api.get_sql_tables()
    
    if "error" in tables_info:
        st.error(f"❌ Error loading database info: {tables_info['error']}")
        return
    
    # Database overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Tables", tables_info.get("total_tables", 0))
    
    with col2:
        db_type = tables_info.get("database_type", "Unknown")
        st.metric("🏢 Database Type", db_type)
    
    with col3:
        if "sample_queries" in tables_info:
            st.metric("📝 Sample Queries", len(tables_info["sample_queries"]))
    
    # Tables list
    tables = tables_info.get("tables", [])
    if tables:
        st.subheader("📋 Available Tables")
        
        # Search and filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 Search tables", placeholder="Enter table name...")
        with col2:
            show_details = st.checkbox("Show Details", value=False)
        
        # Filter tables
        filtered_tables = tables
        if search_term:
            filtered_tables = [
                table for table in tables 
                if search_term.lower() in table.get("table_name", "").lower()
            ]
        
        # Display tables
        if show_details:
            # Detailed view
            for table in filtered_tables:
                with st.expander(f"📊 {table.get('table_name', 'Unknown')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Columns:**")
                        columns = table.get("columns", [])
                        if columns:
                            for col in columns:
                                st.write(f"• {col}")
                        else:
                            st.write("No column information available")
                    
                    with col2:
                        st.write("**Sample Query:**")
                        sample_query = table.get("sample_query", "")
                        if sample_query:
                            st.code(sample_query, language="sql")
                        else:
                            st.write("No sample query available")
        else:
            # Table view
            table_data = []
            for table in filtered_tables:
                table_data.append({
                    "Table Name": table.get("table_name", ""),
                    "Columns": len(table.get("columns", [])),
                    "Database Type": table.get("database_type", "")
                })
            
            if table_data:
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Sample queries (for AdventureWorks)
    sample_queries = tables_info.get("sample_queries", [])
    if sample_queries:
        st.subheader("📝 Sample Queries")
        
        selected_query = st.selectbox(
            "Choose a sample query to try:",
            [""] + sample_queries
        )
        
        if selected_query:
            st.code(selected_query, language="sql")
            
            if st.button("▶️ Try This Query"):
                # Use the chat interface to execute the query
                st.session_state.chat_input = f"Execute this SQL query: {selected_query}"
                st.switch_page("💬 Chat")

def conversation_history():
    """Conversation history management"""
    st.header("📋 Conversation History")
    
    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        user_id_filter = st.text_input("User ID Filter", value=st.session_state.user_id)
    
    with col2:
        limit = st.number_input("Limit", min_value=1, max_value=100, value=20)
    
    with col3:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
    
    # Get conversations
    with st.spinner("Loading conversations..."):
        conversations_data = api.list_conversations(user_id_filter, limit=limit)
    
    if "error" in conversations_data:
        st.error(f"❌ Error: {conversations_data['error']}")
        return
    
    conversations = conversations_data.get("conversations", [])
    
    if not conversations:
        st.info("📭 No conversations found for this user.")
        return
    
    # Display conversations
    st.subheader(f"💬 Conversations ({len(conversations)})")
    
    for i, conv in enumerate(conversations):
        with st.expander(f"Conversation {i+1} - {conv.get('conversation_id', 'Unknown')[:8]}..."):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Created:** {conv.get('created_at', 'Unknown')[:19]}")
                st.write(f"**Updated:** {conv.get('updated_at', 'Unknown')[:19]}")
            
            with col2:
                st.write(f"**Messages:** {conv.get('message_count', 0)}")
                st.write(f"**ID:** `{conv.get('conversation_id', 'Unknown')[:16]}...`")
            
            with col3:
                if st.button(f"🔍 View", key=f"view_{i}"):
                    view_conversation_details(conv.get('conversation_id', ''))
                
                if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                    delete_result = api.delete_conversation(
                        user_id_filter, 
                        conv.get('conversation_id', '')
                    )
                    if "error" not in delete_result:
                        st.success("Conversation deleted!")
                        st.rerun()
                    else:
                        st.error(f"Error: {delete_result['error']}")
            
            # Preview
            preview = conv.get('preview', '')
            if preview:
                st.write(f"**Preview:** {preview}")

def view_conversation_details(conversation_id: str):
    """View detailed conversation"""
    if not conversation_id:
        return
    
    conversation_data = api.get_conversation(st.session_state.user_id, conversation_id)
    
    if "error" in conversation_data:
        st.error(f"Error loading conversation: {conversation_data['error']}")
        return
    
    messages = conversation_data.get("messages", [])
    
    st.subheader(f"📖 Conversation Details: {conversation_id[:16]}...")
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "human":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 User:</strong> {content}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong> {content}
            </div>
            """, unsafe_allow_html=True)

def settings_page():
    """Settings and configuration page"""
    st.header("⚙️ Settings & Configuration")
    
    # API Configuration
    st.subheader("🔗 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_url = st.text_input("API Base URL", value=API_BASE_URL)
        if current_url != API_BASE_URL:
            st.warning("URL change requires app restart to take effect.")
    
    with col2:
        if st.button("🧪 Test Connection"):
            test_api = MultiAgentAPI(current_url)
            health = test_api.health_check()
            
            if "error" not in health:
                st.success("✅ Connection successful!")
            else:
                st.error(f"❌ Connection failed: {health['error']}")
    
    # User Preferences
    st.subheader("👤 User Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_followup = st.checkbox("Auto-show follow-up suggestions", value=True)
        show_sources = st.checkbox("Always show sources", value=True)
        show_images = st.checkbox("Display images in chat", value=True)
    
    with col2:
        default_context = st.text_area(
            "Default Context (JSON)",
            value='{\n  "department": "",\n  "priority": "medium"\n}',
            height=100
        )
        
        try:
            json.loads(default_context)
            st.success("✅ Valid JSON")
        except:
            st.error("❌ Invalid JSON format")
    
    # Export/Import
    st.subheader("📥📤 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Export Conversations**")
        if st.button("📤 Export All Conversations"):
            conversations = api.list_conversations(st.session_state.user_id, limit=1000)
            if "error" not in conversations:
                export_data = {
                    "user_id": st.session_state.user_id,
                    "export_timestamp": datetime.now().isoformat(),
                    "conversations": conversations["conversations"]
                }
                
                st.download_button(
                    label="💾 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"conversations_{st.session_state.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.error("Failed to export conversations")
    
    with col2:
        st.write("**Clear Data**")
        if st.button("🗑️ Clear Session Data", type="secondary"):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.success("Session data cleared!")
        
        if st.button("⚠️ Reset User ID", type="secondary"):
            st.session_state.user_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.success("User ID reset!")
    
    # Debug Information
    st.subheader("🐛 Debug Information")
    
    with st.expander("Session State"):
        debug_state = {
            "user_id": st.session_state.user_id,
            "conversation_id": st.session_state.conversation_id,
            "messages_count": len(st.session_state.messages),
            "system_config": st.session_state.system_config is not None
        }
        st.json(debug_state)
    
    # About
    st.subheader("ℹ️ About")
    st.markdown("""
    **Multi-Agent Conversational AI System**
    
    This Streamlit application provides a user-friendly interface for interacting with the 
    LangGraph-based multi-agent system. Features include:
    
    - 💬 **Chat Interface**: Natural language conversations with AI agents
    - 🔍 **Smart Routing**: Automatic routing to search or SQL agents
    - 📊 **Data Visualization**: Charts and tables for query results
    - 🗃️ **Database Explorer**: Browse available tables and schemas
    - 📋 **History Management**: View"""
    )
    
    
if __name__ == "__main__":
    main()