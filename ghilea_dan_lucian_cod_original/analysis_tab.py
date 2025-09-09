"""
Componenta pentru tab-ul de analiză în aplicația ArtAdvisor.
Versiune completă cu toate funcționalitățile din app.py original și securitate integrată.
"""

import streamlit as st
from PIL import Image
import tempfile
import os
import csv
from datetime import datetime

# Import opțional pentru cropper
try:
    from streamlit_cropper import st_cropper
    CROPPER_AVAILABLE = True
except ImportError:
    CROPPER_AVAILABLE = False
    st_cropper = None

from predict import get_all_predictions, predict_emotions_from_image
from utils.ai_services import generate_narrative_description, synthesize_audio_openai, ask_gpt_about_painting
from utils.data_management import save_analysis_metadata, save_feedback_to_csv
from utils.visualizations import create_emotion_radar_chart
from utils.pdf_generator_advanced import generate_pdf_report, check_pdf_capabilities
from utils.steganography import EmotionalWatermark, calculate_psnr
# Import opțional pentru atacuri adversariale
try:
    from utils.adversarial_attacks import AdversarialAttacks, EmotionModelWrapper
    ADVERSARIAL_AVAILABLE = True
except ImportError:
    ADVERSARIAL_AVAILABLE = False
    AdversarialAttacks = None
    EmotionModelWrapper = None

# Import pentru PIL image processing
from PIL import ImageEnhance, ImageFilter

def render_analysis_tab():
    """Renderează tab-ul de analiză a operelor de artă."""
    
    # Titlu standard
    st.markdown('<h2 class="main-title">Analiză Operă de Artă</h2>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text" style="text-align: center;">Descoperă secretele și emoțiile ascunse în fiecare creație artistică</p>', unsafe_allow_html=True)
    
    # Separator
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    
    # Upload imagine cu design îmbunătățit
    st.markdown("""
    <div style="text-align: center; margin: 1rem 0;">
        <h3 style="color: #D4AF37; margin-bottom: 0.5rem;">Încarcă Opera de Artă</h3>
        <p style="color: rgba(255, 255, 255, 0.7); font-size: 1rem;">
            Selectează o imagine clară a operei pe care dorești să o analizezi
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Formte acceptate: JPG, PNG, JPEG", 
        type=['jpg', 'png', 'jpeg'],
        help="Pentru rezultate optime, folosește imagini cu rezoluție bună și iluminare uniformă",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Afișează imaginea încărcată cu stil modern
        image = Image.open(uploaded_file)
        
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h4 style="color: #D4AF37; margin-bottom: 1rem;">Opera Încărcată</h4>
            <p style="color: rgba(255, 255, 255, 0.7);">
                Poți analiza imaginea completă sau poți izola tabloul pentru o analiză mai precisă.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sistema de cropare și selecție a zonei de analiză
        render_image_cropping_system(image)
        
        # Toggle pentru stilul descrierii narative (ca în app.py)
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h4 style="color: #D4AF37; margin-bottom: 1rem;">Personalizare Interpretare</h4>
            <p style="color: rgba(255, 255, 255, 0.7);">
                Alege stilul în care dorești să fie generată interpretarea narativă.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        style_option = st.selectbox(
            "Stilul descrierii narative:",
            ["Poetic", "Analitic", "Emoțional", "Amuzant", "Istoric"],
            index=0,
            help="Alege tonul în care dorești să fie scrisă interpretarea operei"
        )
        
        # Toggle pentru raport PDF/HTML
        pdf_capabilities = check_pdf_capabilities()
        if pdf_capabilities['weasyprint'] or pdf_capabilities['pdfkit']:
            toggle_text = "Generează și raport PDF pentru download"
            toggle_help = "Activează această opțiune pentru a primi un raport PDF downloadabil cu rezultatele analizei"
        else:
            toggle_text = "Generează și raport HTML pentru download" 
            toggle_help = "Activează această opțiune pentru a primi un raport HTML downloadabil cu rezultatele analizei (convertibil în PDF)"
        
        show_report_toggle = st.toggle(
            toggle_text, 
            value=False,
            help=toggle_help
        )
        
        # Buton pentru analiză cu design modern
        if st.button("Începe Analiza Completă", type="primary", use_container_width=True):
            # Obține imaginea procesată (originală sau cropată)
            cropped_img = get_cropped_image_for_analysis()
            
            if cropped_img is None:
                st.error("Nu s-a putut obține imaginea pentru analiză. Te rog încearcă din nou.")
                return
            
            with st.spinner("Se analizează opera de artă... Acest proces poate dura câteva secunde."):
                try:
                    # Salvează imaginea cropped într-un fișier temporar
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img_file:
                        cropped_img.save(temp_img_file.name)
                        temp_img_path = temp_img_file.name

                    # Progres indicator pentru utilizator
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Pas 1: Obține predicțiile
                    status_text.text("Analizez stilul artistic și autorul...")
                    progress_bar.progress(25)
                    predictions = get_all_predictions(temp_img_path)
                    
                    # Pas 2: Generează descrierea narativă îmbunătățită cu stilul selectat
                    status_text.text("Creez interpretarea narativă în stilul selectat...")
                    progress_bar.progress(50)
                    
                    # Maparea stilului selectat pentru a fi compatibil cu ai_services
                    style_map = {
                        "Poetic": "Poetic",
                        "Analitic": "Analitic", 
                        "Emoțional": "Emoțional",
                        "Amuzant": "Amuzant",
                        "Istoric": "Istoric"
                    }
                    selected_style = style_map.get(style_option, "Poetic")  # Folosește stilul selectat
                    
                    narrative = generate_narrative_description(predictions, selected_style, cropped_img)
                    
                    # Pas 3: Generează audio în română
                    if narrative and narrative != "Nu s-a putut genera descrierea narativă. Vă rugăm să încercați din nou.":
                        status_text.text("Generez narațiunea audio în română...")
                        progress_bar.progress(75)
                        audio_fp = synthesize_audio_openai(narrative)
                        st.session_state['audio'] = audio_fp
                    else:
                        st.session_state['audio'] = None
                    
                    # Pas 4: Finalizare și salvare rezultate
                    status_text.text("Salvez rezultatele și pregătesc raportul...")
                    progress_bar.progress(90)
                    
                    # Salvează rezultatele în session state cu toate datele necesare
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.session_state['analysis_results'] = {
                        'predictions': predictions,
                        'narrative': narrative,
                        'timestamp': timestamp,
                        'style_option': selected_style,
                        'cropped_image': cropped_img,
                        'original_image': image,
                        'show_report_pdf': show_report_toggle
                    }
                    
                    # Salvează metadatele pentru galerie
                    success, img_path, meta_path = save_analysis_metadata(
                        predictions, 
                        narrative, 
                        cropped_img, 
                        timestamp
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Analiza completă!")
                    
                    if success:
                        st.success("Analiza a fost completată cu succes! Interpretarea apare mai jos.")
                        # Auto-scroll către rezultate
                        st.rerun()
                    else:
                        st.warning(f"Analiza s-a finalizat, dar există o problemă la salvare: {meta_path}")
                    
                except Exception as e:
                    st.error(f"A apărut o eroare în timpul analizei: {str(e)}")
                    st.info("Încercați din nou sau contactați suportul tehnic.")
                finally:                    
                    # Șterge fișierul temporar
                    if 'temp_img_path' in locals() and os.path.exists(temp_img_path):
                        os.unlink(temp_img_path)
                    # Șterge progress bar-ul
                    if 'progress_bar' in locals():
                        progress_bar.empty()
                    if 'status_text' in locals():
                        status_text.empty()
    
    # Afișarea rezultatelor cu narațiunea ÎNAINTE de analiză
    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        st.markdown("---")
        
        predictions = st.session_state.analysis_results['predictions']
        narrative = st.session_state.analysis_results['narrative']
        
        # ===== SECȚIUNEA NARATIVĂ - PRIMA ===== 
        if narrative:
            st.markdown("""
            <div class="narrative-section">
                <h2 class="narrative-title">Interpretarea Operei</h2>
                <div class="narrative-content">
                    {}
                </div>
            """.format(narrative), unsafe_allow_html=True)
            
            # Player audio cu funcționalitate îmbunătățită
            if 'audio' in st.session_state and st.session_state.audio:
                st.markdown("""
                <div class="audio-player">
                    <div class="audio-title">Ascultă Interpretarea</div>
                """, unsafe_allow_html=True)
                
                try:
                    # Resetează poziția audio pentru redare corectă
                    st.session_state.audio.seek(0)
                    st.audio(st.session_state.audio, format='audio/mp3', autoplay=False)
                    
                except Exception as e:
                    st.markdown("""
                    <div class="audio-info" style="color: #ff6b6b;">
                        Eroare la redarea audio: {}
                    </div>
                    """.format(str(e)), unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="audio-player">
                    <div class="audio-info">
                        Audio indisponibil pentru această interpretare
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        
        # ===== TITLU PENTRU ANALIZA TEHNICĂ =====
        st.markdown("""
        <h1 class="main-title">Analiza Tehnică Detaliată</h1>
        <h5 class="subtle-text">Rezultatele algoritmilor</h5>
        """, unsafe_allow_html=True)

        # Grid cu 3 coloane pentru top rezultate
        col1, col2, col3 = st.columns(3, gap="large")

        # === COLOANA 1: TOP 3 STILURI ARTISTICE ===
        with col1:
            st.markdown("""
            <h3 class="section-title">Top 3 Stiluri Artistice</h3>
            
            """, unsafe_allow_html=True)
            
            if predictions.get('stil') and predictions['stil'].get('predictions_sorted'):
                styles_data = predictions['stil']['predictions_sorted'][:3]
                
                # Container pentru rezultate cu stil modern
                for i, (style, confidence) in enumerate(styles_data):
                    style_name = style.replace('_', ' ').title()
                    
                    # Card individual pentru fiecare stil cu clasa de rank
                    st.markdown(f"""
                    <div class="result-item progress-rank-{i+1}">
                        <div class="result-header">
                            <span class="result-rank">#{i+1}</span>
                            <span class="result-name">{style_name}</span>
                        </div>
                        <div class="result-confidence">{confidence:.1%} </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar stilizat cu transparență bazată pe ranking
                    st.progress(confidence, text="")
                    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
                
                # Heatmap pentru Stil
                if 'gradcam_image' in predictions.get('stil', {}):
                    st.markdown("""
                    <div class="heatmap-section">
                        <h4>Zona de Interes - Stil</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(predictions['stil']['gradcam_image'], 
                            caption="Heatmap pentru detectarea stilului artistic", 
                            use_container_width=True)
                
                
        # === COLOANA 2: TOP 3 AUTORI PROBABLI ===
        with col2:
            st.markdown("""
                <h3 class="section-title">Top 3 Autori Probabli</h3>
            """, unsafe_allow_html=True)
            
            if predictions.get('autor') and predictions['autor'].get('predictions_sorted'):
                authors_data = predictions['autor']['predictions_sorted'][:3]
                
                # Container pentru rezultate
                for i, (author, confidence) in enumerate(authors_data):
                    author_name = author.replace('_', ' ').title()
                    
                    # Card individual pentru fiecare autor cu clasa de rank
                    st.markdown(f"""
                    <div class="result-item progress-rank-{i+1}">
                        <div class="result-header">
                            <span class="result-rank">#{i+1}</span>
                            <span class="result-name">{author_name}</span>
                        </div>
                        <div class="result-confidence">{confidence:.1%} </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar stilizat cu transparență bazată pe ranking
                    st.progress(confidence, text="")
                    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
                
                # Heatmap pentru Autor
                if 'gradcam_image' in predictions.get('autor', {}):
                    st.markdown("""
                    <div class="heatmap-section">
                        <h4>Zona de Interes - Autor</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(predictions['autor']['gradcam_image'], 
                            caption="Heatmap pentru detectarea autorului", 
                            use_container_width=True)
                
                
        # === COLOANA 3: ANALIZA EMOȚIONALĂ ===
        with col3:
            st.markdown("""
            <h3 class="section-title">Analiza Emoțională</h3>
            """, unsafe_allow_html=True)
            
            if predictions.get('emotie') and predictions['emotie'].get('predictions_sorted'):
                emotions_dict = {emotion: score for emotion, score in predictions['emotie']['predictions_sorted'][:6]}
                
                # Graficul radar pentru emoții
                radar_chart = create_emotion_radar_chart(emotions_dict)
                if radar_chart:
                    st.plotly_chart(radar_chart, use_container_width=True, config={'displayModeBar': False})
                
                # Top 5 emoții
                st.markdown("""
                <div class="emotions-list">
                    <h4>Top 5 Emoții Detectate</h4>
                
                """, unsafe_allow_html=True)
                
                for i, (emotion, score) in enumerate(predictions['emotie']['predictions_sorted'][:5]):
                    emotion_name = emotion.replace('_', ' ').title()
                    st.markdown(f"""
                    <div class="emotion-item">
                        <span class="emotion-rank">#{i+1}</span>
                        <span class="emotion-name">{emotion_name}</span>
                        <span class="emotion-score">{score:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="no-prediction">Nu s-au detectat emoții</div>', unsafe_allow_html=True)

        # ===== SECȚIUNEA DE ÎNTREBĂRI GPT DESPRE PICTURĂ =====
        st.markdown("---")
        st.markdown("""
        <h2 class="main-title">Întreabă despre această operă</h2>
        <p class="subtitle-text" style="text-align: center;">
            Ai întrebări despre culori, tehnici, simboluri sau semnificații? Întreabă un expert!
        </p>
        """, unsafe_allow_html=True)
        
        user_prompt = st.text_input(
            "Scrie întrebarea ta despre pictură:",
            placeholder="Ex: Ce transmite această cromatică? De ce artistul a ales aceste culori? Ce simbolizează elementele din fundal?",
            help="Poți întreba despre orice aspect al operei: tehnici, culori, emoții, simboluri, context istoric, etc."
        )

        if user_prompt and st.button("Trimite întrebarea către Expert", type="primary", use_container_width=True):
            with st.spinner("Expertul analizează întrebarea ta..."):
                gpt_response = ask_gpt_about_painting(user_prompt, predictions)
                if gpt_response:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, rgba(100, 169, 240, 0.1), rgba(212, 175, 55, 0.1)); 
                                 padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(212, 175, 55, 0.3); margin: 1rem 0;">
                        <h4 style="color: #D4AF37; margin-bottom: 1rem;">Răspunsul Expertului:</h4>
                    """, unsafe_allow_html=True)
                    st.write(gpt_response)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error("Nu am putut obține un răspuns. Te rog încearcă din nou.")
        
        # ===== SECȚIUNEA DE FEEDBACK =====
        st.markdown("---")
        st.markdown("""
        <h3 class="section-title">Feedback despre analiză</h3>
        <p style="color: rgba(255, 255, 255, 0.7); text-align: center;">
            Ajută-ne să îmbunătățim sistemul oferind feedback despre corectitudinea analizei
        </p>
        """, unsafe_allow_html=True)
        
        feedback = st.radio(
            "A fost corectă predicția?", 
            ["Da, analiza este corectă", "Nu, analiza are erori", "Nu sunt sigur"],
            horizontal=True,
            index=0
        )

        user_comment = ""
        correct_style = ""
        correct_author = ""

        if feedback == "Nu, analiza are erori":
            user_comment = st.text_area(
                "Ce nu a fost corect? (opțional)", 
                placeholder="Ex: Stilul nu pare corect, nu pare deloc impresionist... Autorul nu poate fi acesta pentru că...",
                help="Comentariile tale ne ajută să îmbunătățim algoritmii de analiză"
            )
            
            col_style, col_author = st.columns(2)
            with col_style:
                correct_style = st.text_input(
                    "Stilul corect ar fi fost:",
                    placeholder="Ex: Romantism, Baroc, etc."
                )
            with col_author:
                correct_author = st.text_input(
                    "Autorul corect ar fi fost:",
                    placeholder="Ex: Van Gogh, Monet, etc."
                )

        if st.button("Trimite feedback", use_container_width=True):
            # Implementarea salvării feedback-ului
            feedback_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": "uploaded_image",  # placeholder
                "stil_prezis": predictions['stil']['predictions_sorted'][0][0] if predictions.get('stil') else "N/A",
                "autor_prezis": predictions['autor']['predictions_sorted'][0][0] if predictions.get('autor') else "N/A",
                "emotii_prezise": ', '.join([e for e, p in predictions['emotie']['predictions_sorted']]) if predictions.get('emotie') else "N/A",
                "feedback": "pozitiv" if feedback == "Da, analiza este corectă" else ("negativ" if feedback == "Nu, analiza are erori" else "neutru"),
                "comentariu_utilizator": user_comment,
                "stil_corect_utilizator": correct_style,
                "autor_corect_utilizator": correct_author
            }
            
            # Salvează feedback-ul în CSV
            save_feedback_to_csv(feedback_data)
            
            if feedback == "Da, analiza este corectă":
                st.success("Mulțumim pentru confirmare! Ne bucurăm că analiza a fost corectă.")
            elif feedback == "Nu, analiza are erori":
                st.success("Feedback salvat! Îți mulțumim pentru observațiile detaliate - ne vor ajuta să îmbunătățim sistemul.")
            else:
                st.info("Mulțumim pentru feedback!")
        
        # ===== OPȚIUNEA DE DESCĂRCARE PDF =====
        if st.session_state.analysis_results.get('show_report_pdf', False):
            st.markdown("---")
            
            # FORȚĂM HTML pentru stabilitate maximă pe Windows
            report_title = "Descărcare Raport HTML"
            report_description = """
            Generează și descarcă un raport complet în format HTML cu analiza acestei opere<br>
            <em style="color: #64A9F0;">Pentru PDF: deschide fișierul HTML în browser → Ctrl+P → "Salvează ca PDF"</em><br>
            <em style="color: #90EE90;">Format recomandat pentru Windows - 100% compatibil</em>
            """
            button_text = "Generează și Descarcă Raport HTML"
            file_extension = "html"
            file_mime = "text/html"
            success_message = "Raportul HTML a fost generat cu succes!"
            info_message = """
            **Pentru a converti în PDF:**
            1. Deschide fișierul HTML în browser (dublu-click)
            2. Apasă **Ctrl+P** 
            3. Selectează **"Salvează ca PDF"**
            4. Alege locația și salvează
            
            Raportul HTML arată identic cu versiunea PDF și se printează perfect!
            """
            
            st.markdown(f"""
            <h3 class="section-title">{report_title}</h3>
            <p style="color: rgba(255, 255, 255, 0.7); text-align: center;">
                {report_description}
            </p>
            """, unsafe_allow_html=True)
            
            if st.button(button_text, type="primary", use_container_width=True):
                with st.spinner(f"Generez raportul {file_extension.upper()}..."):
                    report_buffer = generate_pdf_report(st.session_state.analysis_results)
                    if report_buffer:
                        st.download_button(
                            label=f"Descarcă Raportul {file_extension.upper()}",
                            data=report_buffer,
                            file_name=f"ArtAdvisor_Analiza_{st.session_state.analysis_results['timestamp']}.{file_extension}",
                            mime=file_mime,
                            use_container_width=True
                        )
                        st.success(success_message)
                        st.info(info_message)
        # SECȚIUNEA SECURITATE ȘI AUTENTICITATE - ÎNTOTDEAUNA VIZIBILĂ
        st.markdown("---")
        st.markdown("""
        <h2 class="main-title">Protecție Digitală</h2>
        <p class="subtitle-text" style="text-align: center;">
            Protejează și validează analiza AI a operei tale prin semnătură digitală invizibilă
        </p>
        """, unsafe_allow_html=True)
        
        # Explicație pentru utilizator de ce este importantă
        with st.expander("De ce este utilă semnătura digitală pentru înțelegerea artei?", expanded=False):
            st.markdown("""
            ### Cum te ajută să înțelegi mai bine arta:
            
            **Semnătură Digitală Emoțională:**
            - Protejează descoperirile tale prin salvarea permanentă a emoțiilor detectate în imagine
            - Dovedește autenticitatea analizei tale AI pentru orice verificare viitoare
            - Educativ: Înțelegi cum se poate integra informația în structura unei opere de artă
            - Transparență: Poți verifica oricând ce a interpretat sistemul AI din opera ta
            
            ### Pentru pasionații de artă:
            Această funcție îți oferă o perspectivă tehnică asupra modului în care tehnologia modernă 
            poate proteja și valida analiza artei, dezvoltându-ți capacitatea de observare informată.
            """)
        
        # Verifică dacă există analiză completă
        if 'analysis_results' in st.session_state and st.session_state.analysis_results:
            # Obține datele din analiza curentă
            analysis_results = st.session_state.analysis_results
            predictions = analysis_results.get('predictions', {})
            image = analysis_results.get('cropped_image', analysis_results.get('original_image'))
            
            # Afișează direct funcția de semnătură digitală
            st.markdown("### Semnătură Digitală Emoțională")
            st.markdown("**Protejează și validează analiza ta prin integrarea emoțiilor în structura imaginii**")
            
            render_emotional_watermark_demo(predictions, image)
        else:
            # Mesaj clar când nu există analiză
            st.warning("""
            **Pentru a accesa funcția de protecție digitală, trebuie să analizezi mai întâi o operă de artă!**
            
            **Pașii necesari:**
            1. **Încarcă o imagine** mai sus (JPG, PNG, JPEG)
            2. **Apasă "Începe Analiza Completă"** și așteaptă rezultatele
            3. **Revino aici** pentru funcția de semnătură digitală
            
            **De ce ai nevoie de o analiză mai întâi?**
            - Semnătura digitală integrează emoțiile detectate în opera ta
            - Fără o analiză completă, nu avem emoții pentru a le ascunde în imagine
            - Este ca și cum ai încerca să ascunzi un mesaj fără să ai mesajul
            """)
            
            st.info("""
            **Semnătura Digitală Emoțională** va fi disponibilă după analiză!
            
            **Ce vei putea face:**
            - Ascunzi emoțiile detectate invizibil în imagine
            - Protejezi autenticitatea analizei tale AI
            - Verifici oricând că imaginea conține semnătura ta
            - Descarci imaginea protejată digital
            """)
            
            st.info("""
            **Sfat:** După ce vezi rezultatele analizei mai sus, scroll down și funcția va deveni activă!
            """)

    # Mesaj de încurajare când nu există rezultate - MUTAT LA SFÂRȘIT
    if 'analysis_results' not in st.session_state or not st.session_state.analysis_results:
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0; padding: 3rem; 
                    background: rgba(100, 169, 240, 0.08); border-radius: 20px; 
                    border: 1px solid rgba(100, 169, 240, 0.2);">
            <h3 style="color: #D4AF37; margin-bottom: 1.5rem;">Comenzi pentru ArtAdvisor</h3>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.1rem; line-height: 1.6;">
                <strong>Pas 1:</strong> Încarcă o operă de artă mai sus<br>
                <strong>Pas 2:</strong> Apasă "Începe Analiza Completă"<br>
                <strong>Pas 3:</strong> Vezi rezultatele și accesează funcțiile avansate<br><br>
                <em style="color: #64A9F0;">Toate funcțiile vor deveni active după analiză!</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

def render_emotional_watermark_demo(predictions, image):
    """Demonstrația de semnătură digitală emoțională - pentru protecția și validarea analizelor."""
    
    st.markdown("### Semnătură Digitală Emoțională")
    
    # Explicație clară pentru utilizatori
    st.info("""
    **Funcționalitate:** Integrează emoțiile detectate în structura imaginii ca semnătură invizibilă
    
    **Utilitate practică:** 
    - Salvează permanent interpretarea AI a operei tale
    - Permite validarea autenticității analizei în orice moment viitor
    - Oferă protecție împotriva modificărilor neautorizate
    - Demonstrează transparența procesului de analiză automatizată
    """)
    
    # Initialize watermarking system
    if 'watermark_system' not in st.session_state:
        st.session_state.watermark_system = EmotionalWatermark()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Opera Ta Analizată")
        if image is not None:
            st.image(image, caption="Imaginea care va primi semnătura", use_container_width=True)
        else:
            st.warning("Nu există imagine pentru analiză. Încarcă o operă mai sus.")
            return
        
        # Extract emotions from predictions
        emotions_dict = {}
        if predictions and predictions.get('emotie') and predictions['emotie'].get('predictions_sorted'):
            for emotion, score in predictions['emotie']['predictions_sorted'][:5]:
                emotions_dict[emotion] = score
            
            st.markdown("#### Emoții Detectate (ce va fi ascuns)")
            for emotion, score in emotions_dict.items():
                emotion_name = emotion.replace('_', ' ').title()
                st.write(f"**{emotion_name}**: {score:.1%}")
        else:
            st.warning("Nu s-au detectat emoții în această analiză.")
            return
    
    with col2:
        st.markdown("#### Aplicarea Semnăturii")
        
        st.markdown("""
        **Cum funcționează:**
        1. Emoțiile detectate se convertesc în cod binar
        2. Codul se ascunde în ultimii biți ai pixelilor  
        3. Imaginea arată identic, dar conține informația ascunsă
        4. Doar sistemul nostru poate citi semnătura
        """)
        
        if st.button("Aplică Semnătură Digitală", type="primary", use_container_width=True):
            with st.spinner("Se aplică semnătura digitală invizibilă..."):
                try:
                    # Create metadata
                    metadata = {
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "model_version": "ArtAdvisor_v1.0",
                        "analysis_type": "complet"
                    }
                    
                    # Apply watermark
                    watermarked_image = st.session_state.watermark_system.embed_watermark(
                        image, emotions_dict, metadata
                    )
                    
                    # Calculate quality
                    psnr_value = calculate_psnr(image, watermarked_image)
                    
                    st.session_state.watermarked_image = watermarked_image
                    st.session_state.watermark_metadata = metadata
                    
                    st.success("Semnătură aplicată cu succes!")
                    st.info(f"Calitatea imaginii: {psnr_value:.1f}dB (peste 40dB = perfect invizibil)")
                    
                except Exception as e:
                    st.error(f"Eroare: {str(e)}")
    
    # Show results section
    if 'watermarked_image' in st.session_state:
        st.markdown("---")
        st.markdown("### Rezultatele Semnăturii")
        
        col_orig, col_watermarked = st.columns(2)
        
        with col_orig:
            st.markdown("#### Original")
            st.image(image, caption="Fără semnătură", use_container_width=True)
        
        with col_watermarked:
            st.markdown("#### Cu Semnătură Digitală")
            st.image(st.session_state.watermarked_image, 
                    caption="Arată identic, dar conține emoțiile ascunse", 
                    use_container_width=True)
        
        # Download functionality for watermarked image
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            if st.button("Descarcă Imaginea cu Semnătură", use_container_width=True):
                try:
                    import io
                    
                    # Convert to bytes for download
                    img_buffer = io.BytesIO()
                    st.session_state.watermarked_image.save(img_buffer, format='PNG')
                    img_bytes = img_buffer.getvalue()
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"opera_cu_semnatura_{timestamp}.png"
                    
                    st.download_button(
                        label="Download PNG cu Semnătură",
                        data=img_bytes,
                        file_name=filename,
                        mime="image/png",
                        use_container_width=True
                    )
                    st.success("Imaginea este pregătită pentru download!")
                    
                except Exception as e:
                    st.error(f"Eroare la pregătirea download-ului: {str(e)}")
        
        # Verification section
        st.markdown("### Verificarea Autenticității")
        st.markdown("Testează dacă semnătura digitală este încă intactă:")
        
        if st.button("Verifică Semnătura Ascunsă", use_container_width=True):
            with st.spinner("Se citește semnătura ascunsă..."):
                try:
                    verification = st.session_state.watermark_system.verify_authenticity(
                        st.session_state.watermarked_image
                    )
                    
                    if verification.get("authentic", False):
                        st.success("IMAGINEA ESTE AUTENTICĂ!")
                        st.markdown("**Emoții extrase din semnătura ascunsă:**")
                        
                        emotions = verification.get("emotions", {})
                        for emotion, score in emotions.items():
                            emotion_name = emotion.replace('_', ' ').title()
                            st.write(f"- **{emotion_name}**: {score:.1%}")
                            
                        st.info("**Utilitate:** Poți dovedi oricând că această imagine a fost analizată de AI-ul tău!")
                    else:
                        st.error("Semnătură invalidă sau imagine modificată")
                        st.warning("Imaginea s-ar putea să fi fost modificată după aplicarea semnăturii.")
                        
                except Exception as e:
                    st.error(f"Eroare la verificare: {str(e)}")
        
        # Educational explanation
        with st.expander("Cum mă ajută în viața reală?", expanded=False):
            st.markdown("""
            ### Aplicații Practice:
            
            **Pentru colecționari:**
            - Dovedești autenticitatea analizelor AI ale operei
            - Îți protejezi investiția în artă digitală
            
            **Pentru studenți/cercetători:**
            - Demonstrezi transparența în procesul de analiză
            - Protejezi integritatea academică a lucrărilor
            
            **Pentru artiști:**
            - Verifici cum AI-ul interpretează creațiile tale
            - Protejezi drepturile asupra analizelor digitale
            
            **Pentru curioși:**
            - Înțelegi cum funcționează steganografia în practică
            - Vezi aplicațiile tehnologiei în conservarea artei
            """)

def render_image_cropping_system(image):
    """Renderează sistemul de cropare și detectare automată a tabloului."""
    
    # Setează variabila în session_state dacă nu există
    if 'cropped_img' not in st.session_state:
        st.session_state.cropped_img = image
    if 'use_cropping' not in st.session_state:
        st.session_state.use_cropping = False
    if 'cropping_method' not in st.session_state:
        st.session_state.cropping_method = "manual"
    
    # Secțiunea de opțiuni de cropare
    st.markdown("""
    <div style="background: rgba(100, 169, 240, 0.1); padding: 1.5rem; border-radius: 15px; 
                border: 1px solid rgba(100, 169, 240, 0.3); margin: 1.5rem 0;">
        <h4 style="color: #64A9F0; text-align: center; margin-bottom: 1rem;">
            Optimizare Zonă de Analiză
        </h4>
        <p style="color: rgba(255, 255, 255, 0.8); text-align: center; margin-bottom: 1rem;">
            Pentru analize mai precise, poți izola tabloul din imagine eliminând fundalul, rama sau alte elemente
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Toggle pentru activarea cropării
    use_cropping = st.toggle(
        "Vreau să izol tabloul din imagine",
        value=st.session_state.use_cropping,
        help="Activează această opțiune pentru a selecta doar zona tabloului care va fi analizată"
    )
    st.session_state.use_cropping = use_cropping
    
    if use_cropping:
        # Opțiuni pentru metoda de cropare
        st.markdown("""
        <div style="margin: 1rem 0;">
            <h5 style="color: #D4AF37; margin-bottom: 0.5rem;">Metoda de Selecție:</h5>
        </div>
        """, unsafe_allow_html=True)
        
        cropping_method = st.radio(
            "Cum vrei să selectezi zona tabloului?",
            ["manual", "auto"],
            format_func=lambda x: "Selecție Manuală (Recomandată)" if x == "manual" else "Detectare Automată",
            help="Manual: Ajustezi tu colțurile pentru precizie maximă | Auto: Sistemul încearcă să detecteze tabloul automat",
            horizontal=True
        )
        st.session_state.cropping_method = cropping_method
        
        if cropping_method == "manual":
            render_manual_cropping(image)
        else:
            render_auto_detection(image)
    else:
        # Folosește imaginea originală
        st.session_state.cropped_img = image
        
        # Afișează imaginea originală
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Opera care va fi analizată (imagine completă)", use_container_width=True)

def render_manual_cropping(image):
    """Renderează interfața de cropare manuală."""
    
    st.markdown("""
    <div style="background: rgba(212, 175, 55, 0.1); padding: 1rem; border-radius: 10px; 
                border: 1px solid rgba(212, 175, 55, 0.3); margin: 1rem 0;">
        <h5 style="color: #D4AF37; margin-bottom: 0.5rem;">Instrucțiuni Cropare Manuală:</h5>
        <ul style="color: rgba(255, 255, 255, 0.8); margin: 0; padding-left: 1.5rem;">
            <li>Trage colțurile pentru a selecta doar zona tabloului</li>
            <li>Încearcă să elimini rama, peretele sau alte elemente din fundal</li>
            <li>Asigură-te că incluzi toată suprafața operei de artă</li>
            <li>Pentru rezultate optime, lasă un mic spațiu în jurul tabloului</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Folosește streamlit-cropper dacă este disponibil
        if CROPPER_AVAILABLE:
            cropped_img = st_cropper(
                image, 
                realtime_update=True,
                box_color='#D4AF37',
                aspect_ratio=None,
                return_type='image'
            )
            st.session_state.cropped_img = cropped_img
            
            # Afișează rezultatul cropat
            if cropped_img:
                st.markdown("""
                <div style="text-align: center; margin: 1rem 0;">
                    <h5 style="color: #64A9F0;">Zona Selectată pentru Analiză:</h5>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(cropped_img, caption="Tabloul izolat - gata pentru analiză", use_container_width=True)
        else:
            # Fallback la cropare simplă cu coordonate
            render_coordinate_cropping(image)
            
    except Exception as e:
        st.error(f"Eroare la cropare: {e}")
        render_coordinate_cropping(image)

def render_coordinate_cropping(image):
    """Renderează croparea cu coordonate când streamlit-cropper nu e disponibil."""
    
    st.warning("Sistemul de cropare interactivă nu este disponibil. Folosind metoda alternativă cu coordonate.")
    
    # Afișează imaginea originală pentru referință
    st.image(image, caption="Imagine originală - folosește coordonatele de mai jos pentru a selecta zona", use_container_width=True)
    
    # Coordonate pentru cropare
    col1, col2 = st.columns(2)
    
    with col1:
        left = st.slider("Coordonata X stânga", 0, image.width, 0, help="Marginea stângă a zonei de cropat")
        top = st.slider("Coordonata Y sus", 0, image.height, 0, help="Marginea de sus a zonei de cropat")
    
    with col2:
        right = st.slider("Coordonata X dreapta", 0, image.width, image.width, help="Marginea dreaptă a zonei de cropat")
        bottom = st.slider("Coordonata Y jos", 0, image.height, image.height, help="Marginea de jos a zonei de cropat")
    
    # Validează coordonatele
    if left < right and top < bottom:
        # Crope imaginea
        cropped_img = image.crop((left, top, right, bottom))
        st.session_state.cropped_img = cropped_img
        
        # Afișează rezultatul
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <h5 style="color: #64A9F0;">Zona Selectată pentru Analiză:</h5>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(cropped_img, caption="Tabloul izolat - gata pentru analiză", use_container_width=True)
    else:
        st.error("Coordonatele nu sunt valide. Asigură-te că marginea stângă < dreapta și sus < jos.")

def render_auto_detection(image):
    """Renderează sistemul de detectare automată a tabloului."""
    
    st.markdown("""
    <div style="background: rgba(100, 169, 240, 0.1); padding: 1rem; border-radius: 10px; 
                border: 1px solid rgba(100, 169, 240, 0.3); margin: 1rem 0;">
        <h5 style="color: #64A9F0; margin-bottom: 0.5rem;">Detectare Automată a Tabloului:</h5>
        <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">
            Sistemul va încerca să identifice automat tabloul din imagine folosind detectarea marginilor și contrastului.
            Această metodă funcționează cel mai bine cu tablouri care au contrast clar cu fundalul.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Detectează Tabloul Automat", type="secondary", use_container_width=True):
        with st.spinner("Se detectează automat zona tabloului..."):
            try:
                cropped_img = auto_detect_artwork(image)
                st.session_state.cropped_img = cropped_img
                
                # Afișează rezultatul
                col_orig, col_detected = st.columns(2)
                
                with col_orig:
                    st.markdown("**Imagine Originală:**")
                    st.image(image, use_container_width=True)
                
                with col_detected:
                    st.markdown("**Tablou Detectat:**")
                    st.image(cropped_img, use_container_width=True)
                
                st.success("Tabloul a fost detectat automat! Dacă rezultatul nu este satisfăcător, încearcă selecția manuală.")
                
            except Exception as e:
                st.error(f"Eroare la detectarea automată: {e}")
                st.info("Te rog încearcă selecția manuală pentru rezultate mai bune.")
                st.session_state.cropped_img = image
    else:
        # Afișează imaginea originală până se face detectarea
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Imagine originală - apasă butonul pentru detectare automată", use_container_width=True)
        st.session_state.cropped_img = image

def auto_detect_artwork(image):
    """Detectează automat tabloul din imagine folosind procesare de imagine."""
    
    import cv2
    import numpy as np
    
    # Convertește imaginea PIL la format OpenCV
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        # Convertește RGB la BGR pentru OpenCV
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_cv = img_array
    
    # Convertește la grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Aplicare blur pentru reducerea zgomotului
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectarea marginilor
    edges = cv2.Canny(blurred, 50, 150)
    
    # Găsește contururile
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Găsește cel mai mare contur (probabil tabloul)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Găsește bounding box-ul
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Adaugă un mic padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.width - x, w + 2 * padding)
        h = min(image.height - y, h + 2 * padding)
        
        # Crope imaginea
        cropped_img = image.crop((x, y, x + w, y + h))
        
        # Verifică dacă zona detectată este rezonabilă (nu prea mică)
        if w > image.width * 0.1 and h > image.height * 0.1:
            return cropped_img
    
    # Dacă detectarea automată eșuează, returnează imaginea originală
    return image

def get_cropped_image_for_analysis():
    """Returnează imaginea care trebuie folosită pentru analiză."""
    if 'cropped_img' in st.session_state:
        return st.session_state.cropped_img
    else:
        # Fallback la imaginea din uploaded_file dacă există
        return None
