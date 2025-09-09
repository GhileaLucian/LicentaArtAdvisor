"""
Componenta pentru tab-ul de verificare a semnăturilor digitale în aplicația ArtAdvisor.
Interfață profesională pentru verificarea imaginilor cu semnătură digitală emoțională.
"""

import streamlit as st
from PIL import Image
from datetime import datetime
import tempfile
import os

from utils.steganography import EmotionalWatermark

def render_verification_tab():
    """Renderează tab-ul de verificare a semnăturilor digitale."""
    
    # Header profesional
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #D4AF37; font-size: 2.5rem; margin-bottom: 0.5rem; font-weight: 700;">
            Verificare Semnături Digitale
        </h1>
        <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.2rem; font-weight: 300; margin-bottom: 0;">
            Validează autenticitatea analizelor ArtAdvisor prin tehnologia steganografiei emoționale
        </p>
        <div style="width: 100px; height: 3px; background: linear-gradient(90deg, #D4AF37, #64A9F0); 
                    margin: 1rem auto; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Explicație profesională a funcționalității
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(100, 169, 240, 0.1), rgba(212, 175, 55, 0.1)); 
                     padding: 2rem; border-radius: 20px; border: 1px solid rgba(212, 175, 55, 0.3); 
                     margin: 2rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <h3 style="color: #D4AF37; text-align: center; margin-bottom: 1.5rem; font-weight: 600;">
                🛡️ Tehnologia de Verificare Avansată
            </h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;">
                <div style="text-align: center;">
                    <h4 style="color: #64A9F0; margin-bottom: 0.5rem;">Steganografie</h4>
                    <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">
                        Informațiile sunt ascunse invizibil în structura pixelilor imaginii
                    </p>
                </div>
                <div style="text-align: center;">
                    <h4 style="color: #64A9F0; margin-bottom: 0.5rem;">Emoții Digitale</h4>
                    <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">
                        Rezultatele analizei AI sunt integrate ca semnătură unică
                    </p>
                </div>
                <div style="text-align: center;">
                    <h4 style="color: #64A9F0; margin-bottom: 0.5rem;">Autenticitate</h4>
                    <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">
                        Dovedește că imaginea a fost analizată de sistemul ArtAdvisor
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Secțiunea principală de upload și verificare
    st.markdown("---")
    
    # Container pentru upload cu design elegant
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #D4AF37; margin-bottom: 1rem; font-weight: 600;">
            📤 Încarcă Imaginea pentru Verificare
        </h2>
        <p style="color: rgba(255, 255, 255, 0.7); font-size: 1rem; margin-bottom: 1.5rem;">
            Selectează o imagine care ar putea conține semnătura digitală ArtAdvisor
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload cu stil îmbunătățit
    uploaded_file = st.file_uploader(
        "Drag & Drop sau Click pentru a selecta imaginea",
        type=['jpg', 'png', 'jpeg'],
        key="verification_upload",
        help="Funcționează cu imagini descărcate din secțiunea 'Semnătură Digitală Emoțională'",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Procesează imaginea încărcată
        verification_image = Image.open(uploaded_file)
        
        # Layout profesional cu 2 coloane
        col_image, col_info = st.columns([1.2, 1], gap="large")
        
        with col_image:
            # Container elegant pentru imagine
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px; 
                        border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 1rem;">
                <h4 style="color: #D4AF37; text-align: center; margin-bottom: 1rem;">
                    🖼️ Imaginea de Verificat
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Afișează imaginea cu aspect ratio păstrat
            st.image(
                verification_image, 
                caption=f"📁 {uploaded_file.name}", 
                use_container_width=True
            )
            
            # Informații despre fișier
            file_size = len(uploaded_file.getvalue()) / 1024  # KB
            st.markdown(f"""
            <div style="background: rgba(100, 169, 240, 0.1); padding: 1rem; border-radius: 10px; 
                        margin-top: 1rem; border: 1px solid rgba(100, 169, 240, 0.3);">
                <h5 style="color: #64A9F0; margin-bottom: 0.5rem;">Detalii Fișier</h5>
                <p style="color: rgba(255, 255, 255, 0.8); margin: 0.25rem 0; font-size: 0.9rem;">
                    <strong>Dimensiuni:</strong> {verification_image.size[0]} × {verification_image.size[1]} px
                </p>
                <p style="color: rgba(255, 255, 255, 0.8); margin: 0.25rem 0; font-size: 0.9rem;">
                    <strong>Format:</strong> {verification_image.format}
                </p>
                <p style="color: rgba(255, 255, 255, 0.8); margin: 0.25rem 0; font-size: 0.9rem;">
                    <strong>Mărime:</strong> {file_size:.1f} KB
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_info:
            # Panoul de verificare
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 15px; 
                        border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 1.5rem;">
                <h4 style="color: #D4AF37; text-align: center; margin-bottom: 1rem;">
                    Panoul de Verificare
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Explicație proces
            st.markdown("""
            <div style="background: rgba(212, 175, 55, 0.1); padding: 1.2rem; border-radius: 12px; 
                        border: 1px solid rgba(212, 175, 55, 0.3); margin-bottom: 1.5rem;">
                <h5 style="color: #D4AF37; margin-bottom: 0.8rem;">Procesul de Verificare</h5>
                <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; line-height: 1.6;">
                    <div style="margin: 0.5rem 0; display: flex; align-items: center;">
                        <span style="color: #64A9F0; margin-right: 0.5rem;">1️⃣</span>
                        Scanare automată pentru semnături ascunse
                    </div>
                    <div style="margin: 0.5rem 0; display: flex; align-items: center;">
                        <span style="color: #64A9F0; margin-right: 0.5rem;">2️⃣</span>
                        Extragere metadate și emoții embeddate
                    </div>
                    <div style="margin: 0.5rem 0; display: flex; align-items: center;">
                        <span style="color: #64A9F0; margin-right: 0.5rem;">3️⃣</span>
                        Verificare integritate și autenticitate
                    </div>
                    <div style="margin: 0.5rem 0; display: flex; align-items: center;">
                        <span style="color: #64A9F0; margin-right: 0.5rem;">4️⃣</span>
                        Afișare rezultate detaliate
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Butonul de verificare cu stil
            if st.button("🚀 Începe Verificarea Semnăturii", type="primary", use_container_width=True):
                perform_signature_verification(verification_image, uploaded_file.name)
    
    else:
        # Mesaj când nu e încărcată nicio imagine
        render_empty_state()
    
    # Secțiunea educativă
    render_educational_section()

def perform_signature_verification(image, filename):
    """Performă verificarea semnăturii digitale cu interfață elegantă."""
    
    with st.spinner("Se scanează imaginea pentru semnături digitale..."):
        try:
            # Initialize watermark system dacă nu există
            if 'watermark_system' not in st.session_state:
                st.session_state.watermark_system = EmotionalWatermark()
            
            # Verifică autenticitatea
            verification_result = st.session_state.watermark_system.verify_authenticity(image)
            
            # Container pentru rezultate
            st.markdown("---")
            st.markdown("""
            <h2 style="color: #D4AF37; text-align: center; margin: 2rem 0; font-weight: 600;">
                Rezultatele Verificării
            </h2>
            """, unsafe_allow_html=True)
            
            if verification_result.get("authentic", False):
                render_authentic_result(verification_result, filename)
            else:
                render_non_authentic_result(filename)
                
        except Exception as e:
            render_error_result(str(e))

def render_authentic_result(verification_result, filename):
    """Renderează rezultatul pentru o imagine autentică."""
    
    # Header de succes
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(144, 238, 144, 0.2), rgba(76, 175, 80, 0.1)); 
                 padding: 2rem; border-radius: 20px; border: 2px solid rgba(144, 238, 144, 0.4); 
                 margin: 1.5rem 0; text-align: center; box-shadow: 0 4px 20px rgba(144, 238, 144, 0.2);">
        <h3 style="color: #4CAF50; margin-bottom: 1rem; font-size: 1.8rem;">
            IMAGINE AUTENTICĂ ARTADVISOR
        </h3>
        <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; margin: 0;">
            Această imagine conține semnătura digitală validă și a fost analizată de sistemul nostru AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout cu 2 coloane pentru rezultate
    col_emotions, col_metadata = st.columns(2, gap="large")
    
    with col_emotions:
        # Secțiunea emoții
        emotions = verification_result.get("emotions", {})
        if emotions:
            st.markdown("""
            <div style="background: rgba(100, 169, 240, 0.1); padding: 1.5rem; border-radius: 15px; 
                        border: 1px solid rgba(100, 169, 240, 0.3); margin-bottom: 1rem;">
                <h4 style="color: #64A9F0; text-align: center; margin-bottom: 1rem;">
                    Emoții Din Analiza Originală
                </h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Sortează emoțiile după scor
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            
            for i, (emotion, score) in enumerate(sorted_emotions):
                emotion_name = emotion.replace('_', ' ').title()
                
                # Progress bar stilizat pentru fiecare emoție
                st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                        <span style="color: #D4AF37; font-weight: 600;">#{i+1} {emotion_name}</span>
                        <span style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem;">{score:.1%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(score, text="")
                st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)
    
    with col_metadata:
        # Secțiunea metadata
        metadata = verification_result.get("metadata", {})
        st.markdown("""
        <div style="background: rgba(212, 175, 55, 0.1); padding: 1.5rem; border-radius: 15px; 
                    border: 1px solid rgba(212, 175, 55, 0.3); margin-bottom: 1rem;">
            <h4 style="color: #D4AF37; text-align: center; margin-bottom: 1rem;">
                Detalii Analiză Originală
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        if metadata:
            for key, value in metadata.items():
                if key == "timestamp":
                    # Formatează timestamp-ul
                    try:
                        dt = datetime.strptime(value, "%Y%m%d_%H%M%S")
                        formatted_time = dt.strftime("%d/%m/%Y la %H:%M:%S")
                        render_metadata_item("🕒 Data Analizei", formatted_time)
                    except:
                        render_metadata_item("🕒 Timestamp", value)
                elif key == "model_version":
                    render_metadata_item("🤖 Versiune Model", value)
                elif key == "analysis_type":
                    render_metadata_item("Tip Analiză", value.title())
                else:
                    key_formatted = key.replace('_', ' ').title()
                    render_metadata_item(f"{key_formatted}", str(value))
        else:
            st.markdown("""
            <p style="color: rgba(255, 255, 255, 0.6); text-align: center; font-style: italic;">
                Nu sunt disponibile metadate suplimentare
            </p>
            """, unsafe_allow_html=True)
    
    # Secțiunea de acțiuni
    render_action_buttons_authentic(filename)

def render_metadata_item(label, value):
    """Renderează un item de metadata cu stil elegant."""
    st.markdown(f"""
    <div style="background: rgba(255, 255, 255, 0.05); padding: 0.8rem; border-radius: 8px; 
                margin-bottom: 0.8rem; border: 1px solid rgba(255, 255, 255, 0.1);">
        <div style="color: #64A9F0; font-size: 0.85rem; margin-bottom: 0.2rem;">{label}</div>
        <div style="color: rgba(255, 255, 255, 0.9); font-weight: 500;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def render_action_buttons_authentic(filename):
    """Renderează butoanele de acțiune pentru imagini autentice."""
    
    st.markdown("---")
    st.markdown("""
    <h3 style="color: #D4AF37; text-align: center; margin: 2rem 0;">
        Ce Poți Face Acum
    </h3>
    """, unsafe_allow_html=True)
    
    col_action1, col_action2, col_action3 = st.columns(3, gap="medium")
    
    with col_action1:
        if st.button("Analizează Din Nou", use_container_width=True):
            st.info("**Sugestie**: Mergi la tab-ul 'Analiză Operă' și încarcă această imagine pentru o analiză completă actualizată!")
    
    with col_action2:
        if st.button("📈 Compară Analize", use_container_width=True):
            st.info("**Tip**: Poți compara emoțiile din semnătură cu o analiză nouă pentru a vedea consistența!")
    
    with col_action3:
        if st.button("📤 Verifică Alta", use_container_width=True):
            st.rerun()

def render_non_authentic_result(filename):
    """Renderează rezultatul pentru o imagine neautentică."""
    
    # Header de avertizare
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(255, 165, 0, 0.2), rgba(255, 193, 7, 0.1)); 
                 padding: 2rem; border-radius: 20px; border: 2px solid rgba(255, 165, 0, 0.4); 
                 margin: 1.5rem 0; text-align: center; box-shadow: 0 4px 20px rgba(255, 165, 0, 0.2);">
        <h3 style="color: #FF8C00; margin-bottom: 1rem; font-size: 1.8rem;">
            NICIO SEMNĂTURĂ ARTADVISOR GĂSITĂ
        </h3>
        <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; margin: 0;">
            Această imagine nu conține semnătura digitală ArtAdvisor sau a fost modificată
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Explicații posibile
    col_reasons, col_actions = st.columns(2, gap="large")
    
    with col_reasons:
        st.markdown("""
        <div style="background: rgba(255, 165, 0, 0.1); padding: 1.5rem; border-radius: 15px; 
                    border: 1px solid rgba(255, 165, 0, 0.3); margin-bottom: 1rem;">
            <h4 style="color: #FF8C00; margin-bottom: 1rem;">Posibile Motive</h4>
            <div style="color: rgba(255, 255, 255, 0.8); line-height: 1.8;">
                <div style="margin: 0.5rem 0;">Imaginea nu a fost analizată cu ArtAdvisor</div>
                <div style="margin: 0.5rem 0;">Imaginea a fost modificată după aplicarea semnăturii</div>
                <div style="margin: 0.5rem 0;">📉 Imaginea a fost compresată prea mult</div>
                <div style="margin: 0.5rem 0;">🔧 Semnătura a fost deteriorată în procesul de salvare</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_actions:
        st.markdown("""
        <div style="background: rgba(100, 169, 240, 0.1); padding: 1.5rem; border-radius: 15px; 
                    border: 1px solid rgba(100, 169, 240, 0.3); margin-bottom: 1rem;">
            <h4 style="color: #64A9F0; margin-bottom: 1rem;">Ce Poți Face</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Analizează Acum", use_container_width=True):
            st.info("Mergi la tab-ul 'Analiză Operă' pentru a analiza această imagine și a-i aplica semnătura!")
        
        st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
        
        if st.button("Încearcă Altă Imagine", use_container_width=True):
            st.rerun()

def render_error_result(error_message):
    """Renderează rezultatul în caz de eroare."""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(255, 99, 71, 0.2), rgba(220, 20, 60, 0.1)); 
                 padding: 2rem; border-radius: 20px; border: 2px solid rgba(255, 99, 71, 0.4); 
                 margin: 1.5rem 0; text-align: center;">
        <h3 style="color: #FF6347; margin-bottom: 1rem;">Eroare la Verificare</h3>
        <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">
            A apărut o problemă în timpul verificării semnăturii
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📝 Detalii Eroare", expanded=False):
        st.code(error_message)
        
        st.markdown("""
        **Posibile cauze:**
        - Format de imagine necompatibil
        - Imagine deteriorată sau coruptă
        - Problemă tehnică temporară
        - Dimensiunea imaginii este prea mare
        """)
    
    if st.button("Încearcă Din Nou", use_container_width=True):
        st.rerun()

def render_empty_state():
    """Renderează starea când nu e încărcată nicio imagine."""
    
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0; padding: 3rem; 
                background: rgba(100, 169, 240, 0.05); border-radius: 25px; 
                border: 2px dashed rgba(100, 169, 240, 0.3);">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📤</div>
        <h3 style="color: #64A9F0; margin-bottom: 1.5rem; font-weight: 600;">
            Încarcă o Imagine pentru Verificare
        </h3>
        <p style="color: rgba(255, 255, 255, 0.7); font-size: 1.1rem; margin-bottom: 1.5rem; max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.6;">
            Selectează o imagine care ar putea conține semnătura digitală ArtAdvisor pentru a verifica autenticitatea și a recupera analiza originală
        </p>
        <div style="background: rgba(212, 175, 55, 0.1); padding: 1rem; border-radius: 12px; 
                    border: 1px solid rgba(212, 175, 55, 0.3); max-width: 500px; margin: 0 auto;">
            <p style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem; margin: 0;">
                <strong>Tip:</strong> Funcționează cu imagini descărcate din secțiunea "Semnătură Digitală Emoțională"
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_educational_section():
    """Renderează secțiunea educativă despre verificarea semnăturilor."""
    
    st.markdown("---")
    st.markdown("""
    <h2 style="color: #D4AF37; text-align: center; margin: 2rem 0; font-weight: 600;">
        📚 Cum Funcționează Verificarea
    </h2>
    """, unsafe_allow_html=True)
    
    # Tutorial în expander
    with st.expander("🎓 Ghid Complet de Utilizare", expanded=False):
        col_tutorial1, col_tutorial2 = st.columns(2)
        
        with col_tutorial1:
            st.markdown("""
            ### Pașii pentru Verificare:
            
            **1. Obține o imagine cu semnătură:**
            - Analizează o operă în tab-ul "Analiză Operă"
            - Aplică "Semnătură Digitală Emoțională"
            - Descarcă imaginea cu semnătură
            
            **2. Verifică imaginea:**
            - Întoarce-te aici și încarcă imaginea descărcată
            - Apasă "Începe Verificarea Semnăturii"
            - Vezi rezultatele automate în timp real
            
            **3. Înțelege rezultatele:**
            - = Imagine autentică cu analiza originală
            - = Fără semnătură sau modificată
            """)
        
        with col_tutorial2:
            st.markdown("""
            ### De ce e Utilă Această Funcție:
            
            **Pentru Autenticitate:**
            - Demonstrezi că analiza e legitimă
            - Verifici integritatea imaginilor
            - Protejezi împotriva falsificării
            
            **Pentru Recuperare:**
            - Recuperezi analizele din imagini vechi
            - Vezi emoțiile detectate anterior
            - Accesezi metadatele originale
            
            **Pentru Educație:**
            - Înveți despre tehnologiile de protecție digitală
            - Înțelegi steganografia în practică
            - Descoperi aplicațiile în conservarea artei
            """)
    
    # Cards cu informații tehnice
    col_tech1, col_tech2, col_tech3 = st.columns(3, gap="medium")
    
    with col_tech1:
        st.markdown("""
        <div style="background: rgba(100, 169, 240, 0.1); padding: 1.5rem; border-radius: 15px; 
                    border: 1px solid rgba(100, 169, 240, 0.3); text-align: center; height: 200px; 
                    display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔒</div>
            <h4 style="color: #64A9F0; margin-bottom: 0.8rem;">Securitate</h4>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; margin: 0;">
                Tehnologie criptografică avansată pentru protecția datelor
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_tech2:
        st.markdown("""
        <div style="background: rgba(212, 175, 55, 0.1); padding: 1.5rem; border-radius: 15px; 
                    border: 1px solid rgba(212, 175, 55, 0.3); text-align: center; height: 200px; 
                    display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">👁️</div>
            <h4 style="color: #D4AF37; margin-bottom: 0.8rem;">Invizibilitate</h4>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; margin: 0;">
                Modificări imperceptibile care nu afectează calitatea vizuală
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_tech3:
        st.markdown("""
        <div style="background: rgba(144, 238, 144, 0.1); padding: 1.5rem; border-radius: 15px; 
                    border: 1px solid rgba(144, 238, 144, 0.3); text-align: center; height: 200px; 
                    display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚡</div>
            <h4 style="color: #90EE90; margin-bottom: 0.8rem;">Rapiditate</h4>
            <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; margin: 0;">
                Verificare instantanee cu rezultate în timp real
            </p>
        </div>
        """, unsafe_allow_html=True)
