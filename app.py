# app.py
# Main Streamlit Application - This is what the user sees and interacts with
# Run with: streamlit run app.py

import streamlit as st
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our custom modules
from rag_pipeline import initialize_rag, retrieve_context
from agents import parser_agent, router_agent, solver_agent, verifier_agent, explainer_agent
from memory import save_to_memory, update_feedback, find_similar_problems, get_memory_stats
from ocr_handler import extract_text_from_image, preprocess_math_text
from audio_handler import transcribe_audio, fix_math_speech

# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🧮 Math Mentor - JEE Helper",
    page_icon="🧮",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS for better looks
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .agent-box {
        background: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px 15px;
        border-radius: 5px;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px 15px;
        border-radius: 5px;
    }
    .hitl-box {
        background: #ffd7d7;
        border-left: 4px solid #dc3545;
        padding: 10px 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INITIALIZE RAG (only once, cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading knowledge base...")
def load_rag():
    """Load the RAG pipeline once and cache it"""
    return initialize_rag("knowledge_base")


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    
    # Title
    st.markdown('<div class="main-title">🧮 Math Mentor</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#666;">AI-powered JEE Math Solver with RAG + Multi-Agent System</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Load RAG
    vector_store = load_rag()
    
    # ─────────────────────────────────────────────
    # SIDEBAR - Stats and Memory
    # ─────────────────────────────────────────────
    with st.sidebar:
        st.header("📊 Memory Stats")
        stats = get_memory_stats()
        
        col1, col2 = st.columns(2)
        col1.metric("Total Solved", stats["total"])
        col2.metric("Correct ✅", stats["correct"])
        
        if stats["topics"]:
            st.subheader("Topics Covered")
            for topic, count in stats["topics"].items():
                st.write(f"• {topic.capitalize()}: {count}")
        
        st.divider()
        st.caption("Built with Groq + LangChain + FAISS")
        st.caption("💡 Free tools only!")
    
    # ─────────────────────────────────────────────
    # INPUT MODE SELECTOR
    # ─────────────────────────────────────────────
    st.subheader("📥 Choose Input Method")
    input_mode = st.radio(
        "How do you want to enter the math problem?",
        ["⌨️ Type", "📸 Image Upload", "🎤 Audio Upload"],
        horizontal=True
    )
    
    raw_text = ""
    extracted_text = ""
    show_hitl = False
    hitl_reason = ""
    
    # ─────────────────────────────────────────────
    # A) TEXT INPUT
    # ─────────────────────────────────────────────
    if input_mode == "⌨️ Type":
        raw_text = st.text_area(
            "Enter your math problem:",
            placeholder="e.g. Find the roots of x^2 - 5x + 6 = 0",
            height=100
        )
    
    # ─────────────────────────────────────────────
    # B) IMAGE INPUT
    # ─────────────────────────────────────────────
    elif input_mode == "📸 Image Upload":
        uploaded_image = st.file_uploader(
            "Upload a photo or screenshot of your math problem",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", width=400)
            
            with st.spinner("🔍 Reading text from image..."):
                ocr_result = extract_text_from_image(uploaded_image)
            
            if ocr_result.get("error"):
                st.error(f"OCR Error: {ocr_result['error']}")
                raw_text = st.text_area(
                    "Please type the problem manually:",
                    height=80
                )
            else:
                # Show extracted text for user to verify
                st.markdown("**📝 Extracted Text** (you can edit if needed):")
                extracted_text = preprocess_math_text(ocr_result["text"])
                raw_text = st.text_area(
                    f"OCR extracted (confidence: {ocr_result['confidence']*100:.0f}%)",
                    value=extracted_text,
                    height=80
                )
                
                # HITL trigger if low confidence
                if ocr_result.get("low_confidence"):
                    show_hitl = True
                    hitl_reason = f"OCR confidence is low ({ocr_result['confidence']*100:.0f}%). Please verify the extracted text above."
    
    # ─────────────────────────────────────────────
    # C) AUDIO INPUT
    # ─────────────────────────────────────────────
    elif input_mode == "🎤 Audio Upload":
        uploaded_audio = st.file_uploader(
            "Upload audio of you speaking the math problem",
            type=["mp3", "wav", "m4a", "ogg"]
        )
        
        if uploaded_audio:
            st.audio(uploaded_audio)
            
            with st.spinner("🎙️ Transcribing audio..."):
                transcript_result = transcribe_audio(uploaded_audio)
            
            if transcript_result.get("error"):
                st.error(f"Audio Error: {transcript_result['error']}")
                raw_text = st.text_area("Please type the problem manually:", height=80)
            else:
                # Fix math speech phrases
                fixed_text = fix_math_speech(transcript_result["text"])
                
                st.markdown("**📝 Transcript** (you can edit if needed):")
                raw_text = st.text_area(
                    f"Whisper transcript (confidence: {transcript_result['confidence']*100:.0f}%)",
                    value=fixed_text,
                    height=80
                )
                
                # HITL trigger if unclear
                if transcript_result.get("needs_confirmation"):
                    show_hitl = True
                    hitl_reason = "Transcription may be unclear. Please verify the text above."
    
    # ─────────────────────────────────────────────
    # HITL WARNING - Show before solving
    # ─────────────────────────────────────────────
    if show_hitl and raw_text:
        st.markdown(f"""
        <div class="hitl-box">
            ⚠️ <b>Human Review Needed:</b> {hitl_reason}<br>
            Please check and correct the text above before clicking Solve.
        </div>
        """, unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────
    # SOLVE BUTTON
    # ─────────────────────────────────────────────
    st.divider()
    
    if st.button("🚀 Solve Problem", type="primary", disabled=not raw_text.strip()):
        if not raw_text.strip():
            st.warning("Please enter a math problem first!")
        else:
            solve_problem(raw_text, vector_store)


# ─────────────────────────────────────────────
# CORE SOLVING FUNCTION
# Runs all 5 agents in sequence
# ─────────────────────────────────────────────
def solve_problem(raw_text: str, vector_store):
    """
    Main pipeline:
    raw_text → Parser → Router → RAG → Solver → Verifier → Explainer
    """
    
    st.header("🔍 Solution Process")
    
    # Check for similar past problems in memory
    st.markdown("**🧠 Checking memory for similar problems...**")
    similar = find_similar_problems(raw_text)
    if similar:
        with st.expander(f"📚 Found {len(similar)} similar past problem(s) in memory"):
            for s in similar:
                st.write(f"**Past problem:** {s['parsed_problem'].get('problem_text', '')}")
                st.write(f"**Past answer:** {s['solution'].get('solution', '')}")
                st.write(f"**Feedback:** {s.get('user_feedback', 'not rated')}")
                st.divider()
    
    # ── AGENT TRACE (shows what each agent is doing) ──
    agent_trace_container = st.container()
    
    with agent_trace_container:
        st.markdown("### 🤖 Agent Trace")
        
        # ─── AGENT 1: PARSER ───
        with st.spinner("🔄 Agent 1: Parsing problem..."):
            parsed = parser_agent(raw_text)
        
        with st.expander("✅ Parser Agent", expanded=False):
            st.markdown('<div class="agent-box">', unsafe_allow_html=True)
            st.write("**Topic:**", parsed.get("topic", "unknown").capitalize())
            st.write("**Variables:**", ", ".join(parsed.get("variables", [])) or "none")
            st.write("**Constraints:**", ", ".join(parsed.get("constraints", [])) or "none")
            st.write("**Clean Problem:**", parsed.get("problem_text", raw_text))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # HITL: Parser found ambiguity
        if parsed.get("needs_clarification"):
            st.markdown(f"""
            <div class="hitl-box">
                ⚠️ <b>Parser needs clarification:</b> {parsed.get('clarification_reason', '')}
            </div>
            """, unsafe_allow_html=True)
            
            clarification = st.text_input(
                "Please clarify the problem:",
                key="clarification_input"
            )
            if clarification:
                raw_text = raw_text + " " + clarification
                parsed = parser_agent(raw_text)
        
        # ─── AGENT 2: ROUTER ───
        with st.spinner("🔄 Agent 2: Planning solution strategy..."):
            routing = router_agent(parsed)
        
        with st.expander("✅ Router Agent", expanded=False):
            st.markdown('<div class="agent-box">', unsafe_allow_html=True)
            st.write("**Problem Type:**", routing.get("problem_type", ""))
            st.write("**Difficulty:**", routing.get("difficulty", ""))
            st.write("**Strategy:**", routing.get("strategy", ""))
            st.write("**Tools:**", ", ".join(routing.get("tools_needed", [])))
            st.write("**Estimated Steps:**", routing.get("estimated_steps", 0))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ─── RAG RETRIEVAL ───
        problem_text = parsed.get("problem_text", raw_text)
        
        with st.spinner("🔄 RAG: Searching knowledge base..."):
            context, sources = retrieve_context(
                vector_store, 
                problem_text + " " + parsed.get("topic", "")
            )
        
        with st.expander("📚 Retrieved Knowledge", expanded=False):
            if sources:
                st.write("**Sources used:**", ", ".join(sources))
            if context:
                st.text_area("Retrieved context:", context, height=150, disabled=True)
            else:
                st.warning("No specific context retrieved from knowledge base")
        
        # ─── AGENT 3: SOLVER ───
        with st.spinner("🔄 Agent 3: Solving the problem..."):
            solution = solver_agent(
                problem_text,
                routing.get("strategy", ""),
                context
            )
        
        with st.expander("✅ Solver Agent", expanded=False):
            st.markdown('<div class="agent-box">', unsafe_allow_html=True)
            st.write("**Confidence:**", f"{solution.get('confidence', 0)*100:.0f}%")
            st.write("**Formulas Used:**", ", ".join(solution.get("formulas_used", [])))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ─── AGENT 4: VERIFIER ───
        with st.spinner("🔄 Agent 4: Verifying solution..."):
            verification = verifier_agent(problem_text, solution)
        
        with st.expander("✅ Verifier Agent", expanded=False):
            status = "✅ Correct" if verification.get("is_correct") else "❌ Issues Found"
            st.write(f"**Status:** {status}")
            st.write("**Confidence:**", f"{verification.get('confidence', 0)*100:.0f}%")
            if verification.get("issues_found"):
                st.write("**Issues:**", verification["issues_found"])
            if verification.get("corrections"):
                st.write("**Corrections:**", verification["corrections"])
        
        # HITL: Verifier not confident
        if verification.get("needs_human_review"):
            st.markdown(f"""
            <div class="hitl-box">
                ⚠️ <b>Verifier requests human review:</b> {verification.get('review_reason', 'Low confidence in solution')}
            </div>
            """, unsafe_allow_html=True)
        
        # ─── AGENT 5: EXPLAINER ───
        with st.spinner("🔄 Agent 5: Writing explanation..."):
            explanation = explainer_agent(problem_text, solution)
    
    # ─────────────────────────────────────────────
    # FINAL OUTPUT - The main result
    # ─────────────────────────────────────────────
    st.divider()
    st.header("📋 Answer")
    
    # Confidence bar
    confidence = solution.get("confidence", 0.5)
    conf_color = "#28a745" if confidence >= 0.8 else "#ffc107" if confidence >= 0.6 else "#dc3545"
    st.markdown(f"""
        <div style="margin-bottom:10px;">
            <b>Confidence:</b> 
            <span style="color:{conf_color}; font-size:1.2rem; font-weight:bold;">
                {confidence*100:.0f}%
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    # Steps
    st.markdown("**🪜 Solution Steps:**")
    for i, step in enumerate(solution.get("steps", []), 1):
        st.markdown(f"**{i}.** {step}")
    
    # Final Answer
    st.markdown(f"""
    <div class="success-box">
        <b>🎯 Final Answer:</b><br>
        <span style="font-size:1.3rem;">{solution.get('solution', 'Could not solve')}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Explanation
    st.markdown("**💡 Explanation:**")
    st.info(explanation)
    
    # ─────────────────────────────────────────────
    # FEEDBACK BUTTONS
    # ─────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Was this solution correct?")
    
    # Save to memory first
    memory_id = save_to_memory(
        original_input=raw_text,
        parsed_problem=parsed,
        solution=solution,
        explanation=explanation,
        verifier_result=verification
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Correct! Good job", type="primary", key="feedback_correct"):
            update_feedback(memory_id, "correct")
            st.success("🎉 Great! This solution has been saved as correct.")
            st.balloons()
    
    with col2:
        incorrect_comment = st.text_input(
            "If incorrect, explain what's wrong:",
            key="incorrect_comment"
        )
        if st.button("❌ Incorrect", key="feedback_incorrect"):
            update_feedback(memory_id, "incorrect", incorrect_comment)
            st.error("📝 Noted! This will help improve future solutions.")
            if incorrect_comment:
                st.write("Your feedback:", incorrect_comment)


# ─────────────────────────────────────────────
# RUN THE APP
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
