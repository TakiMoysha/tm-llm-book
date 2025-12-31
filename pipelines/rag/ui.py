import streamlit as st

from pipelines.rag.search_records import upload_database_to_qdrant


# === Настройки страницы ===
st.set_page_config(
    page_title="RAG Product Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Стили ===
st.markdown("""
    <style>
        .stButton>button { margin-top: 24px; }
        .stDataFrame { font-size: 14px; }
        .stAlert { font-size: 16px; }
    </style>
""", unsafe_allow_html=True)

# === Заголовок ===
st.title("🔍 RAG Product Search Engine")
st.markdown("Search products by natural language queries with manufacturer filtering.")

# === Боковая панель ===
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Database")
    if st.button("🔄 Reload Database into Qdrant"):
        with st.spinner("Uploading records to Qdrant..."):
            status = upload_database_to_qdrant()
        st.success(status)
    
    st.subheader("Filters")
    manufacturers = get_all_manufacturers()
    selected_manufacturer = st.selectbox(
        "Filter by Manufacturer",
        options=["All"] + manufacturers,
        index=0
    )
    
    st.markdown("---")
    st.caption("Powered by Qdrant + OpenAI Embeddings")

# === Основной контент ===
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., 'durable tennis racket for beginners'",
        key="search_input"
    )

with col2:
    st.write("")  # пустая строка для выравнивания
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

# === Результаты ===
if search_button and query.strip():
    with st.spinner("Searching..."):
        manufacturer = None if selected_manufacturer == "All" else selected_manufacturer
        results = search_products(query=query, manufacturer_filter=manufacturer, top_k=5)
    
    if results:
        st.subheader(f"✅ Found {len(results)} results")
        st.dataframe(
            data=results,
            column_config={
                "product_name": st.column_config.TextColumn("Product Name", width="large"),
                "product_description": st.column_config.TextColumn("Description", width="medium"),
                "technical_specs": st.column_config.TextColumn("Technical Specs", width="medium"),
                "manufacturer": st.column_config.TextColumn("Manufacturer", width="small"),
                "score": st.column_config.NumberColumn("Similarity Score", format="%.4f")
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("No results found. Try a different query or reload the database.")

elif search_button and not query.strip():
    st.warning("Please enter a search query.")

# === Статус ===
if st.session_state.get("last_status"):
    st.info(st.session_state.last_status)

# === Футер ===
st.markdown("---")
st.caption("Built with Streamlit + LangChain + Qdrant")


