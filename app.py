import streamlit as st
import os
import re
from inference import predict
from download_report import generate_report_pdf

# st.title("AI vs Human text Classifier", anchor=False)
st.set_page_config(page_title="Report Generator", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<h1 style='text-align: center;'>AI vs Human text Classifier</h1>", unsafe_allow_html=True)
st.divider()

MAX_WORDS = 1000

def enforce_word_limit():
    text = st.session_state.get("input_text", "")
    word_count = len(text.split())
    if word_count > MAX_WORDS:
        # Truncate to max words
        words = text.split()
        st.session_state.input_text = " ".join(words[:MAX_WORDS])

def extract_text_from_file(uploaded_file):
    """Extract text from an uploaded .doc, .docx, or .pdf file."""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    text = ""
    
    if file_extension == ".pdf":
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except ImportError:
            st.error("PyPDF2 is required to read PDF files. Install it with: pip install PyPDF2")
            return ""
    elif file_extension == ".docx":
        try:
            import docx
            doc = docx.Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except ImportError:
            st.error("python-docx is required to read Word files. Install it with: pip install python-docx")
            return ""
    
    # Limit to MAX_WORDS
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
    
    return text

def on_file_upload():
    """Callback when a file is uploaded. Populates the text area with extracted content."""
    uploaded_file = st.session_state.get("file_uploader")
    if uploaded_file is not None:
        extracted_text = extract_text_from_file(uploaded_file)
        if extracted_text:
            st.session_state.input_text = extracted_text

def generate_text_stats(input_text):
    """Generate statistics for the input text."""
    words = input_text.split()
    word_count = len(words)
    
    # Sentence length distribution (split by . ! ?)
    sentences = re.split(r'[.!?]+', input_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_lengths = [len(s.split()) for s in sentences]
    
    stats_lines = [f"**Word Count:** {word_count}"]
    stats_lines.append("")
    stats_lines.append("**Sentence Length Distribution:**")
    if sentence_lengths:
        avg_sent_len = sum(sentence_lengths) / len(sentence_lengths)
        stats_lines.append(f"- Number of sentences: {len(sentences)}")
        stats_lines.append(f"- Average sentence length: {avg_sent_len:.1f} words")
        stats_lines.append(f"- Shortest sentence: {min(sentence_lengths)} words")
        stats_lines.append(f"- Longest sentence: {max(sentence_lengths)} words")
        stats_lines.append("")
        stats_lines.append("| Sentence # | Word Count |")
        stats_lines.append("|------------|------------|")
        for i, slen in enumerate(sentence_lengths, 1):
            stats_lines.append(f"| {i} | {slen} |")
    else:
        stats_lines.append("- No sentences detected.")
    
    return "\n".join(stats_lines)

def generate_prediction_response(input_text, selected_model):
    """Run inference and format the response."""
    # Load model (already cached in inference.py)
    label, proba = predict(input_text, selected_model)
    
    label_str = "AI-generated" if label == 1 else "Human-written"
    confidence = proba if label == 1 else 1 - proba
    
    lines = [f"**Model:** {selected_model}"]
    lines.append("")
    lines.append(f"**Prediction:** {label_str}")
    lines.append(f"**Confidence:** {confidence:.4f} ({confidence:.2%})")
    
    return "\n".join(lines)

left_col, right_col = st.columns([0.38, 0.62], gap="large")

with left_col:
	# st.markdown('<div class="section-label">Title</div>', unsafe_allow_html=True)
	st.text("Paste your text here or Upload a file")
	st.text_area("", value="", height=150, placeholder="text field", label_visibility="collapsed", key="input_text", on_change=enforce_word_limit)
	word_count = len(st.session_state.get("input_text", "").split())
	if word_count > 0:
		if word_count > MAX_WORDS:
			st.warning(f"Word limit exceeded! Maximum {MAX_WORDS} words allowed. (Current: {word_count})")
		else:
			st.caption(f"Word count: {word_count}/{MAX_WORDS}")
    
	st.markdown("<p style='text-align: center; margin: 8px 0;'><b>OR</b></p>", unsafe_allow_html=True)

	st.file_uploader("Upload a file (.docx, .pdf)", type=["docx", "pdf"], accept_multiple_files=False, label_visibility="collapsed", key="file_uploader", on_change=on_file_upload)

	st.divider()

	model_col, dropdown_col = st.columns([0.8, 0.2], vertical_alignment="center")
	with model_col:
		model_files = [os.path.splitext(f)[0] for f in os.listdir("models") if f.endswith((".pkl", ".h5")) and os.path.splitext(f)[0] != "tfidf_vectorizer"]
		model_options = ["pick model here"] + sorted(model_files) + ["all models for comparison"]
		st.selectbox("", model_options, label_visibility="collapsed", key="selected_model")
	

	st.divider()

	if st.button("Generate report", use_container_width=False, type="primary"):
		input_text = st.session_state.get("input_text", "").strip()
		selected_model = st.session_state.get("selected_model", "pick model here")
		
		if not input_text:
			st.warning("Please enter some text or upload a file first.")
		else:
			# Generate text statistics
			stats_text = generate_text_stats(input_text)
			
			# Generate prediction response
			if selected_model == "pick model here":
				response_text = "**No model selected.** Please choose a model from the dropdown."
			elif selected_model == "all models for comparison":
				# Run all models and compare (exclude tfidf_vectorizer since it's not a classifier)
				model_names = [m for m in model_files if m != "tfidf_vectorizer"]
				response_lines = ["**All Models Comparison:**", ""]
				response_lines.append("| Model | Prediction | Confidence |")
				response_lines.append("|-------|------------|------------|")
				for mname in model_names:
					try:
						label, proba = predict(input_text, mname)
						label_str = "AI" if label == 1 else "Human"
						conf = proba if label == 1 else 1 - proba
						response_lines.append(f"| {mname} | {label_str} | {conf:.2%} |")
					except Exception as e:
						response_lines.append(f"| {mname} | Error | {str(e)} |")
				response_text = "\n".join(response_lines)
			else:
				response_text = generate_prediction_response(input_text, selected_model)
			
			st.session_state.report_stats = stats_text
			st.session_state.report_response = response_text

with right_col:
	text_stats = st.container(border = True)
	response = st.container(border = True)
	
	has_report = bool(st.session_state.get("report_stats")) and bool(st.session_state.get("report_response"))
	download_left, download_right = st.columns([0.64, 0.36])
	with download_right:
		if has_report:
			st.download_button(
				"Download report",
				data=generate_report_pdf(
					st.session_state.report_stats,
					st.session_state.report_response,
				),
				file_name="report.pdf",
				mime="application/pdf",
				use_container_width=True,
			)
		else:
			st.button("Download report", disabled=True, use_container_width=True)
	
with text_stats:
    report_stats = st.session_state.get("report_stats", "")
    if report_stats:
        st.markdown(report_stats)
    else:
        st.markdown("*No statistics yet.*")

with response:
    report_response = st.session_state.get("report_response", "")
    if report_response:
        st.markdown(report_response)
    else:
        st.markdown('*No response yet.*')
	
	

st.markdown("</div>", unsafe_allow_html=True)
