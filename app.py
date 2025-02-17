import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.config import config
from src.data_loader import DocumentLoader
from src.rag_engine import RAGEngine
from src.analytics import SentimentAnalyzer


# Page configuration
st.set_page_config(
    page_title="FinRAG - Financial Document Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "documents" not in st.session_state:
        st.session_state.documents = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    if "sentiment_analyzer" not in st.session_state:
        st.session_state.sentiment_analyzer = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "sentiment_results" not in st.session_state:
        st.session_state.sentiment_results = None


def render_sidebar() -> None:
    """Render the sidebar with file upload and settings."""
    with st.sidebar:
        st.title("📊 FinRAG")
        st.markdown("---")

        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=config.openai_api_key,
            help="Enter your OpenAI API key"
        )

        if api_key:
            config.openai_api_key = api_key

        st.markdown("---")

        # File uploader
        st.subheader("📁 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a financial document (e.g., SEC 10-K filing)"
        )

        if uploaded_file is not None:
            if st.button("Process Document", type="primary", use_container_width=True):
                process_document(uploaded_file)

        st.markdown("---")

        # Document status
        if st.session_state.chunks:
            st.success(f"✅ Document loaded: {len(st.session_state.chunks)} chunks")
        else:
            st.info("No document uploaded yet")

        st.markdown("---")
        st.caption("Built for BNY Mellon Data Science Internship")


def process_document(uploaded_file) -> None:
    """Process the uploaded PDF document."""
    with st.spinner("Processing document..."):
        try:
            # Load and chunk document
            loader = DocumentLoader()
            chunks = loader.process_pdf(
                file_bytes=uploaded_file.read(),
                filename=uploaded_file.name
            )
            st.session_state.chunks = chunks

            # Initialize RAG engine
            if not config.openai_api_key:
                st.error("Please enter your OpenAI API key")
                return

            rag_engine = RAGEngine(openai_api_key=config.openai_api_key)
            rag_engine.initialize(chunks)
            st.session_state.rag_engine = rag_engine

            # Initialize sentiment analyzer
            with st.spinner("Loading sentiment model..."):
                st.session_state.sentiment_analyzer = SentimentAnalyzer()

            # Clear previous results
            st.session_state.chat_history = []
            st.session_state.sentiment_results = None

            st.success("Document processed successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")


def render_chat_interface() -> None:
    """Render the chat interface for RAG Q&A."""
    st.header("Chat with Document")

    if not st.session_state.rag_engine:
        st.info("Please upload a document to start chatting")
        return

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_engine.query(prompt)
                    answer = response["result"]
                    st.markdown(answer)

                    # Show sources
                    if response.get("source_documents"):
                        with st.expander("📚 Sources"):
                            for i, doc in enumerate(response["source_documents"][:3]):
                                st.markdown(f"**Chunk {i+1}:**")
                                st.caption(doc.page_content[:300] + "...")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


def render_sentiment_dashboard() -> None:
    """Render the sentiment analysis dashboard."""
    st.header("Sentiment Dashboard")

    if not st.session_state.chunks:
        st.info("Please upload a document to analyze sentiment")
        return

    if not st.session_state.sentiment_analyzer:
        st.warning("Sentiment analyzer not initialized")
        return

    # Analyze button
    if st.session_state.sentiment_results is None:
        if st.button("Analyze Document Sentiment", type="primary"):
            analyze_sentiment()
        return

    results = st.session_state.sentiment_results

    # Overall sentiment
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Positive",
            f"{results['overall']['positive']:.1%}",
            delta=None
        )
    with col2:
        st.metric(
            "Negative",
            f"{results['overall']['negative']:.1%}",
            delta=None
        )
    with col3:
        st.metric(
            "Neutral",
            f"{results['overall']['neutral']:.1%}",
            delta=None
        )

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Pie chart
        fig_pie = px.pie(
            values=list(results["distribution"].values()),
            names=list(results["distribution"].keys()),
            title="Sentiment Distribution (by chunk count)",
            color_discrete_map={
                "positive": "#2ecc71",
                "negative": "#e74c3c",
                "neutral": "#95a5a6"
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Bar chart
        fig_bar = go.Figure(data=[
            go.Bar(
                x=["Positive", "Negative", "Neutral"],
                y=[
                    results["overall"]["positive"],
                    results["overall"]["negative"],
                    results["overall"]["neutral"]
                ],
                marker_color=["#2ecc71", "#e74c3c", "#95a5a6"]
            )
        ])
        fig_bar.update_layout(
            title="Average Sentiment Scores",
            yaxis_title="Probability",
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed analysis table
    st.subheader("Chunk-by-Chunk Analysis")
    if results.get("dataframe") is not None:
        df = results["dataframe"]
        st.dataframe(
            df[["chunk_id", "dominant_sentiment", "positive", "negative", "neutral"]],
            use_container_width=True,
            hide_index=True
        )

    # Section analysis
    st.subheader("Section Analysis")
    section = st.selectbox(
        "Analyze specific section:",
        ["Risk Factors", "Management Discussion", "Financial Condition"]
    )

    if st.button("Analyze Section"):
        analyze_section(section)


def analyze_sentiment() -> None:
    """Perform sentiment analysis on the document."""
    with st.spinner("Analyzing sentiment (this may take a moment)..."):
        try:
            analyzer = st.session_state.sentiment_analyzer
            chunks_text = [chunk.page_content for chunk in st.session_state.chunks]

            # Get overall sentiment
            overall = analyzer.get_document_sentiment(chunks_text)

            # Get distribution
            distribution = analyzer.get_sentiment_distribution(chunks_text)

            # Get detailed dataframe
            df = analyzer.analyze_chunks(chunks_text)

            st.session_state.sentiment_results = {
                "overall": overall,
                "distribution": distribution,
                "dataframe": df
            }

            st.rerun()

        except Exception as e:
            st.error(f"Error analyzing sentiment: {str(e)}")


def analyze_section(section_name: str) -> None:
    """Analyze sentiment of a specific section."""
    keywords_map = {
        "Risk Factors": ["risk factors", "risks", "risk factor"],
        "Management Discussion": ["management's discussion", "md&a", "management discussion"],
        "Financial Condition": ["financial condition", "liquidity", "capital resources"]
    }

    full_text = " ".join([chunk.page_content for chunk in st.session_state.chunks])
    keywords = keywords_map.get(section_name, [section_name.lower()])

    result = st.session_state.sentiment_analyzer.analyze_section(full_text, keywords)

    if result:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{section_name} - Positive", f"{result['positive']:.1%}")
        with col2:
            st.metric(f"{section_name} - Negative", f"{result['negative']:.1%}")
        with col3:
            st.metric(f"{section_name} - Neutral", f"{result['neutral']:.1%}")
    else:
        st.warning(f"Could not find '{section_name}' section in the document")


def main() -> None:
    """Main application entry point."""
    initialize_session_state()
    render_sidebar()

    # Main content tabs
    tab1, tab2 = st.tabs(["Chat", "Sentiment Analysis"])

    with tab1:
        render_chat_interface()

    with tab2:
        render_sentiment_dashboard()


if __name__ == "__main__":
    main()
