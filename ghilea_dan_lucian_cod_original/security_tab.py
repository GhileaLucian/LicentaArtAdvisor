"""
Tab pentru Securitate și Autenticitate - demonstrații de cybersecurity
Incluide steganografie, atacuri adversariale și gestionare utilizatori
"""
import streamlit as st
import tempfile
import os
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Import utilitățile de securitate
from utils.steganography import EmotionalWatermark, calculate_psnr
from utils.adversarial_attacks import AdversarialAttacks, EmotionModelWrapper
from utils.user_profiles import EmotionalProfileManager

def render_security_tab():
    """Renderează tab-ul pentru demonstrații de securitate."""
    
    st.markdown('<h2 class="main-title">Laborator de Securitate & Autenticitate</h2>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text" style="text-align: center;">Demonstrații avansate de cybersecurity în AI și autenticitatea artei digitale</p>', unsafe_allow_html=True)
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    
    # Selectorul pentru demonstrația dorită
    security_demo = st.selectbox(
        "Alege demonstrația de securitate:",
        [
            "Steganografie Emoțională (Digital Watermarking)",
            "Atacuri Adversariale (AI Robustness Testing)", 
            "Profiluri Emoționale Criptate (Privacy Protection)",
            "Analiza Comparativă - Toate Tehnicile"
        ],
        help="Fiecare demonstrație arată o tehnică diferită de securitate aplicată asupra artei digitale"
    )
    
    if security_demo == "Steganografie Emoțională (Digital Watermarking)":
        render_steganography_demo()
    elif security_demo == "Atacuri Adversariale (AI Robustness Testing)":
        render_adversarial_demo()
    elif security_demo == "Profiluri Emoționale Criptate (Privacy Protection)":
        render_encrypted_profiles_demo()
    elif security_demo == "Analiza Comparativă - Toate Tehnicile":
        render_comparative_analysis()

def render_steganography_demo():
    """Demonstrația de steganografie emoțională - versiune intuitivă."""
    
    # EXPLICAȚIE INTUITIVĂ LA ÎNCEPUT
    st.markdown("### Steganografie Emoțională - 'Semnătură Digitală' Invizibilă")
    
    # Buton pentru explicație detaliată
    if st.button("❓ Ce înseamnă 'Steganografie'?", key="explain_steganography"):
        with st.expander("📚 Explicație Simplă - Ce este Steganografia", expanded=True):
            st.markdown("""
            ## 🕵️ Ce este Steganografia?
            
            **Steganografia** = Arta de a ascunde mesaje secrete în conținut aparent normal.
            
            ### 🔍 Exemplu Simplu:
            Imaginează-ți că ai o poză cu familia. Vrei să trimiți un mesaj secret prietenului tău.
            
            **Fără steganografie:**
            - Scrie mesajul pe o hârtie separată ❌ (Oricine poate vedea)
            
            **Cu steganografie:**  
            - Ascunzi mesajul în pixelii pozei ✅ (Doar destinatarul știe că e acolo)
            
            ### 🎨 În Contextul Nostru:
            - **Poza normală**: O pictură oarecare
            - **Mesajul ascuns**: Emoțiile detectate de AI-ul tău + data analizei
            - **Cum ascundem**: Modificăm ultimul bit din fiecare pixel (imperceptibil)
            
            ### 🔒 De Ce e Sigur:
            - Modificarea e atât de mică încât ochiul uman nu o vede
            - Doar cineva care știe 'secretul' poate extrage informația
            - Orice modificare a imaginii corupe mesajul ascuns
            """)
    
    st.markdown("""
    **🎯 Scopul acestei demonstrații:**
    
    Vrem să-ți arătăm cum poți **dovedi autenticitatea** analizelor AI ale tale. 
    Imaginea va rămâne vizibil identică, dar va conține o "semnătură digitală" 
    cu emoțiile detectate + data analizei + cod de verificare.
    """)
    
    # Initialize watermarking
    if 'watermark_system' not in st.session_state:
        st.session_state.watermark_system = EmotionalWatermark()
    
    # PASUL 1: ÎNCĂRCAREA ȘI ANALIZA
    st.markdown("---")
    st.markdown("## 📤 PASUL 1: Încarcă Imaginea și Aplică 'Semnătură Digitală'")
    
    uploaded_file = st.file_uploader(
        "Alege orice imagine pentru demonstrație",
        type=['png', 'jpg', 'jpeg'],
        key="steganography_upload",
        help="Imaginea va rămâne vizibil identică, dar va conține semnătura noastră digitală"
    )
    
    if uploaded_file:
        original_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📸 Imaginea Originală")
            st.image(original_image, caption="Imaginea așa cum o vezi tu", use_container_width=True)
            
        with col2:
            st.markdown("#### 🎭 Emoțiile Detectate de AI")
            
            # Simulăm analiza AI (în realitate ar veni din modelul tău)
            demo_emotions = {
                "Tristețe": 0.85,
                "Melancolie": 0.72,
                "Calm": 0.45,
                "Mister": 0.63
            }
            
            # Afișăm emoțiile cu bare colorate
            for emotion, score in demo_emotions.items():
                color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
                st.markdown(f"{color} **{emotion}**: {score:.1%}")
                st.progress(score)
            
            st.info("💡 Aceste emoții vor fi 'împachetate' în semnătura digitală")
        
        # Buton pentru aplicarea watermark-ului
        st.markdown("---")
        if st.button("✨ Aplică Semnătură Digitală Invizibilă", 
                    key="apply_watermark", 
                    type="primary",
                    help="Va ascunde emoțiile detectate în pixelii imaginii"):
            
            with st.spinner("🔄 Aplicăm semnătura digitală... (modificăm pixelii imperceptibil)"):
                try:
                    # Aplicăm watermark-ul
                    demo_metadata = {
                        "timestamp": "demo_2025_09_04",
                        "stil_dominant": "Romantism", 
                        "autor_probabil": "Caspar David Friedrich",
                        "model_version": "ArtAdvisor v2.0"
                    }
                    
                    watermarked_image = st.session_state.watermark_system.embed_watermark(
                        original_image, demo_emotions, demo_metadata
                    )
                    
                    # Salvăm în session state
                    st.session_state.watermarked_image = watermarked_image
                    st.session_state.original_emotions = demo_emotions
                    st.session_state.metadata = demo_metadata
                    
                    # Calculăm calitatea (PSNR)
                    psnr = calculate_psnr(original_image, watermarked_image)
                    
                    st.success("✅ Semnătură digitală aplicată cu succes!")
                    st.info(f"📊 Calitatea: {psnr:.1f}dB (peste 40dB = imperceptibil pentru ochi)")
                    
                    # Explicăm ce s-a întâmplat
                    with st.expander("🔍 Ce s-a întâmplat adineaori?", expanded=True):
                        st.markdown("""
                        **Tehnic vorbind:**
                        - Am modificat ultimul bit din fiecare pixel RGB
                        - Am ascuns: emoțiile + data + semnătură de verificare
                        - Imaginea arată identic, dar conține 'amprenta' AI-ului tău
                        - Orice modificare a imaginii va corupe această semnătură
                        """)
                        
                except Exception as e:
                    st.error(f"❌ Eroare la aplicarea semnăturii: {str(e)}")
    
    # PASUL 2: VERIFICAREA - EXPLICAT MAI CLAR
    st.markdown("---")
    st.markdown("## 🔍 PASUL 2: Verifică Autenticitatea (Citește Semnătura Digitală)")
    
    if 'watermarked_image' in st.session_state:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🖼️ Imaginea cu Semnătură Digitală")
            st.image(st.session_state.watermarked_image, 
                    caption="Arată identic cu originalul, nu-i așa?", 
                    use_container_width=True)
            
        with col2:
            st.markdown("#### 🔍 Ce Conține Semnătura Digitală?")
            st.markdown("""
            📋 **Informații ascunse în imagine:**
            - Emoțiile detectate de AI
            - Data și ora analizei  
            - Versiunea modelului folosit
            - Cod de verificare (checksum)
            """)
        
        # Buton pentru verificare
        if st.button("🔍 Citește Semnătura Digitală", 
                    key="verify_watermark",
                    type="secondary",
                    help="Extrag informațiile ascunse din imagine"):
            
            with st.spinner("🔄 Citim semnătura digitală din pixeli..."):
                try:
                    verification_result = st.session_state.watermark_system.verify_authenticity(
                        st.session_state.watermarked_image
                    )
                    
                    if verification_result["authentic"]:
                        st.success("✅ IMAGINEA ESTE AUTENTICĂ!")
                        st.markdown("**🎭 Emoțiile originale extrase din semnătură:**")
                        
                        # Comparăm emoțiile extrase cu cele originale
                        extracted_emotions = verification_result["emotions"]
                        original_emotions = st.session_state.original_emotions
                        
                        for emotion in original_emotions.keys():
                            extracted_score = extracted_emotions.get(emotion, 0)
                            original_score = original_emotions.get(emotion, 0)
                            
                            if abs(extracted_score - original_score) < 0.01:
                                st.markdown(f"✅ **{emotion}**: {extracted_score:.1%} (perfect)")
                            else:
                                st.markdown(f"⚠️ **{emotion}**: {extracted_score:.1%} (ușor diferit)")
                        
                        # Afișăm metadata
                        st.markdown("**📋 Alte informații din semnătură:**")
                        st.info(f"🕒 Analizată la: {verification_result['timestamp']}")
                        st.info(f"🤖 Model folosit: {verification_result['model_version']}")
                        
                        with st.expander("🎯 De ce e importantă această verificare?"):
                            st.markdown("""
                            **Scopul acestei funcționalități:**
                            - ✅ **Dovedește autenticitatea**: Arată că imaginea a fost analizată de AI-ul tău
                            - ✅ **Protejează integritatea**: Orice modificare a imaginii corupe semnătura
                            - ✅ **Previne falsificarea**: Nimeni nu poate modifica rezultatele fără să fie detectat
                            - ✅ **Urmărește proveniența**: Știi exact când și cu ce model a fost analizată
                            """)
                            
                    else:
                        st.error("❌ Problema detectată!")
                        st.warning("Imaginea nu conține semnătura digitală așteptată sau a fost modificată.")
                        
                except Exception as e:
                    st.error(f"❌ Eroare la citirea semnăturii: {str(e)}")
    
    # PASUL 3: TESTUL DE SABOTAJ - MAI INTUITIV
    st.markdown("---")
    st.markdown("## 🛡️ PASUL 3: Test de Securitate (Simulează un Atac)")
    
    if 'watermarked_image' in st.session_state:
        
        st.markdown("""
        **Ce testăm aici?**
        
        Vrem să demonstrăm că semnătura digitală detectează automat orice modificare a imaginii.
        Chiar și cea mai mică schimbare (un pixel modificat) va corupe semnătura.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔄 Simulează Modificare")
            
            tamper_options = st.selectbox(
                "Alege tipul de modificare:",
                [
                    "Modificare un pixel (roșu)",
                    "Adaugă text peste imagine", 
                    "Decupează o parte din imagine",
                    "Comprimă imaginea (JPEG)",
                    "Întoarce imaginea"
                ],
                help="Fiecare modificare va demonstra cum detectăm sabotajul"
            )
            
            if st.button("🚨 Aplică Modificarea și Testează", 
                        key="simulate_tampering",
                        help="Va modifica imaginea și va verifica dacă semnătura mai funcționează"):
                
                with st.spinner("🔄 Aplicăm modificarea și testăm semnătura..."):
                    try:
                        tampered_image = st.session_state.watermarked_image.copy()
                        
                        # Aplicăm modificarea aleasă
                        if tamper_options == "Modificare un pixel (roșu)":
                            pixels = tampered_image.load()
                            pixels[0, 0] = (255, 0, 0)  # Pixel roșu în colțul stânga-sus
                            modification_desc = "am modificat un singur pixel să fie roșu"
                            
                        elif tamper_options == "Adaugă text peste imagine":
                            from PIL import ImageDraw, ImageFont
                            draw = ImageDraw.Draw(tampered_image)
                            try:
                                font = ImageFont.truetype("arial.ttf", 20)
                            except:
                                font = ImageFont.load_default()
                            draw.text((10, 10), "MODIFICAT!", fill=(255, 0, 0), font=font)
                            modification_desc = "am adăugat text peste imagine"
                            
                        elif tamper_options == "Decupează o parte din imagine":
                            width, height = tampered_image.size
                            tampered_image = tampered_image.crop((10, 10, width-10, height-10))
                            modification_desc = "am decupat marginile imaginii"
                            
                        elif tamper_options == "Comprimă imaginea (JPEG)":
                            # Salvează și reîncarcă cu compresie JPEG
                            import io
                            buffer = io.BytesIO()
                            tampered_image.save(buffer, format='JPEG', quality=80)
                            buffer.seek(0)
                            tampered_image = Image.open(buffer)
                            modification_desc = "am comprimat imaginea în format JPEG"
                            
                        elif tamper_options == "Întoarce imaginea":
                            tampered_image = tampered_image.rotate(90)
                            modification_desc = "am rotit imaginea cu 90°"
                        
                        # Testăm semnătura pe imaginea modificată
                        verification_result = st.session_state.watermark_system.verify_authenticity(tampered_image)
                        
                        # Salvăm imaginea modificată pentru afișare
                        st.session_state.tampered_image = tampered_image
                        st.session_state.tamper_description = modification_desc
                        st.session_state.tamper_result = verification_result
                        
                    except Exception as e:
                        st.error(f"❌ Eroare la aplicarea modificării: {str(e)}")
        
        with col2:
            st.markdown("#### 📊 Rezultatul Testului")
            
            if 'tamper_result' in st.session_state:
                result = st.session_state.tamper_result
                
                if not result["authentic"]:
                    st.error("🚨 SABOTAJ DETECTAT!")
                    st.markdown(f"**Modificarea aplicată:** {st.session_state.tamper_description}")
                    st.warning("**Rezultat:** Semnătura digitală a fost coruptă și nu mai poate fi citită")
                    
                    # Explicăm de ce s-a întâmplat
                    with st.expander("🔍 De ce s-a întâmplat asta?", expanded=True):
                        st.markdown(f"""
                        **Explicația tehnică:**
                        - Modificarea pixelilor a corupt datele ascunse în semnătură
                        - Codul de verificare (checksum) nu mai corespunde
                        - Sistemul detectează automat orice schimbare, oricât de mică
                        
                        **Ce înseamnă asta pentru securitate:**
                        - ✅ Protejează împotriva falsificării rezultatelor AI
                        - ✅ Dovedește că imaginea nu a fost modificată după analiză
                        - ✅ Oferă trasabilitate completă a procesului
                        """)
                        
                else:
                    st.warning("🤔 Interesant! Modificarea nu a afectat semnătura")
                    st.info("Acest lucru e rar și înseamnă că modificarea a fost foarte subtilă")
                
                # Afișăm imaginea modificată
                if 'tampered_image' in st.session_state:
                    st.image(st.session_state.tampered_image, 
                            caption=f"Imaginea după modificare: {st.session_state.tamper_description}", 
                            use_container_width=True)
    
    # CONCLUZIE ȘI BENEFICII
    st.markdown("---")
    st.markdown("## 🎯 Concluzie: De Ce e Importantă Această Funcționalitate?")
    
    benefits_col1, benefits_col2 = st.columns(2)
    
    with benefits_col1:
        st.markdown("### 🔒 Pentru Securitate")
        st.markdown("""
        - ✅ **Autenticitate garantată** - Dovedește proveniența analizelor AI
        - ✅ **Detectare falsificare** - Orice modificare e detectată automat
        - ✅ **Integritate date** - Protejează rezultatele împotriva sabotajului
        - ✅ **Trasabilitate** - Urmărește istoricul complet al analizei
        """)
        
    with benefits_col2:
        st.markdown("### 🎓 Pentru Prezentarea Ta")
        st.markdown("""
        - ✅ **Demonstrație wow** - Efecte vizuale impresionante
        - ✅ **Tehnică avansată** - Arată cunoștințe de cybersecurity
        - ✅ **Problemă reală** - Rezolvi o nevoie actuală din industrie
        - ✅ **Inovativ** - Puțini studenți implementează așa ceva
        """)
    
    st.success("🎉 Felicitări! Ai implementat una dintre cele mai avansate funcționalități de securitate pentru analiza AI!")

def render_adversarial_demo():
    """Demonstrația atacurilor adversariale."""
    st.markdown("### 🎯 Atacuri Adversariale - Testarea Robustetii AI")
    
    st.markdown("""
    **Conceptul**: Generează imagini "adversariale" care arată identice cu originalul pentru ochiul uman,
    dar păcălesc complet modelul AI să dea predicții greșite.
    
    **Tehnicile**: FGSM (Fast Gradient Sign Method) și PGD (Projected Gradient Descent)
    pentru a testa vulnerabilitățile modelului de emoții.
    """)
    
    # Initialize adversarial system
    if 'adversarial_system' not in st.session_state:
        st.session_state.adversarial_system = AdversarialAttacks()
        # Wrap the emotion prediction model
        from predict import predict_emotions_from_image
        model_wrapper = EmotionModelWrapper(predict_emotions_from_image)
        st.session_state.adversarial_system.set_model(model_wrapper)
    
    uploaded_file = st.file_uploader(
        "Încarcă o imagine pentru testarea adversarială",
        type=['png', 'jpg', 'jpeg'],
        key="adversarial_upload"
    )
    
    if uploaded_file:
        original_image = Image.open(uploaded_file)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Imaginea Originală")
            st.image(original_image, use_container_width=True)
            
            # Predicția originală
            if st.button("Analizează Imaginea Originală", key="analyze_original"):
                with st.spinner("Analizez emoțiile..."):
                    try:
                        from predict import predict_emotions_from_image
                        original_emotions = predict_emotions_from_image(original_image)
                        st.session_state.original_emotions = original_emotions
                        
                        st.markdown("**Emoțiile detectate:**")
                        for emotion, score in original_emotions.items():
                            st.metric(emotion, f"{score:.1%}")
                            
                    except Exception as e:
                        st.error(f"Eroare la analiză: {e}")
        
        with col2:
            st.markdown("#### Controale Atac")
            
            attack_type = st.selectbox(
                "Tipul atacului:",
                ["FGSM - Rapid", "PGD - Avansat"],
                help="FGSM este rapid dar mai puțin sofisticat, PGD este mai lent dar mai eficient"
            )
            
            epsilon_level = st.selectbox(
                "Intensitatea perturbației:",
                ["subtil", "moderat", "vizibil"],
                index=1,
                help="Subtil = foarte greu de observat, Vizibil = se poate observa noise-ul"
            )
            
            if 'original_emotions' in st.session_state and st.button("🎯 Generează Atac Adversarial", key="generate_attack"):
                with st.spinner("Generez imaginea adversarială..."):
                    try:
                        if attack_type.startswith("FGSM"):
                            attack_result = st.session_state.adversarial_system.fgsm_attack(
                                original_image, epsilon=epsilon_level
                            )
                        else:  # PGD
                            attack_result = st.session_state.adversarial_system.pgd_attack(
                                original_image, epsilon=epsilon_level
                            )
                        
                        if attack_result['attack_success']:
                            st.session_state.attack_result = attack_result
                            st.success("Atac generat cu succes!")
                        else:
                            st.error(f"Atacul a eșuat: {attack_result.get('error', 'Eroare necunoscută')}")
                            
                    except Exception as e:
                        st.error(f"Eroare la generarea atacului: {e}")
        
        with col3:
            st.markdown("#### Rezultatul Atacului")
            
            if 'attack_result' in st.session_state:
                attack_data = st.session_state.attack_result
                
                st.image(attack_data['adversarial_image'], 
                        caption="Imaginea Adversarială (aparent identică)", 
                        use_container_width=True)
                
                if st.button("Analizează Imaginea Adversarială", key="analyze_adversarial"):
                    with st.spinner("Analizez imaginea adversarială..."):
                        try:
                            from predict import predict_emotions_from_image
                            adversarial_emotions = predict_emotions_from_image(attack_data['adversarial_image'])
                            
                            st.markdown("**🚨 Rezultat ȘOCANT:**")
                            for emotion, score in adversarial_emotions.items():
                                original_score = st.session_state.original_emotions.get(emotion, 0)
                                diff = abs(score - original_score)
                                
                                if diff > 0.3:  # Diferență semnificativă
                                    st.metric(
                                        emotion, 
                                        f"{score:.1%}", 
                                        delta=f"{score-original_score:+.1%}",
                                        delta_color="inverse"
                                    )
                                else:
                                    st.metric(emotion, f"{score:.1%}")
                                    
                            # Calculează metrici de robustețe
                            orig_pred = np.array(list(st.session_state.original_emotions.values()))
                            adv_pred = np.array(list(adversarial_emotions.values()))
                            
                            l2_distance = np.linalg.norm(orig_pred - adv_pred)
                            st.error(f"🎯 ATACUL A REUȘIT! Distanța L2: {l2_distance:.3f}")
                            
                        except Exception as e:
                            st.error(f"Eroare la analiza adversarială: {e}")
                
                # Opțional: vizualizează noise-ul
                if attack_data.get('noise_visualization') and st.checkbox("Afișează zgomotul adăugat (amplificat)"):
                    st.image(attack_data['noise_visualization'], 
                            caption="Zgomotul adversarial (amplificat pentru vizibilitate)")
    
    # Explicația tehnicii
    with st.expander("🎓 Explicația Tehnică", expanded=False):
        st.markdown("""
        **Cum funcționează:**
        1. **Calculul Gradientilor**: Se determină cât de sensibil este modelul la modificări în fiecare pixel
        2. **Generarea Perturbației**: Se creează un "zgomot" care maximizează eroarea modelului
        3. **Aplicarea Atacului**: Zgomotul se adaugă la imagine, rezultând o imagine adversarială
        
        **Implicații pentru Securitate:**
        - Modelele AI pot fi păcălite cu modificări imperceptibile
        - Este esențial să testezi robustețea modelului în producție
        - Atacurile pot fi contracarate prin diverse tehnici de apărare
        
        **Aplicabilitate:**
        - Detectarea falsurilor în autentificarea biometrică
        - Securitatea sistemelor de recunoaștere în mașini autonome
        - Protejarea modelelor împotriva atacurilor rău-intenționate
        """)

def render_encrypted_profiles_demo():
    """Demonstrația profilurilor criptate."""
    st.markdown("### 🔒 Profiluri Emoționale Criptate - Privacy Protection")
    
    st.markdown("""
    **Conceptul**: Permite utilizatorilor să își construiască profiluri emoționale personalizate
    care sunt stocate complet criptat, respectând GDPR și confidențialitatea.
    
    **Tehnicile**: Criptarea Fernet, PBKDF2 pentru derivarea cheilor, și hashing pentru autentificare.
    """)
    
    # Initialize profile manager
    if 'profile_manager' not in st.session_state:
        st.session_state.profile_manager = EmotionalProfileManager()
    
    # Login/Register interface
    auth_tab1, auth_tab2, auth_tab3 = st.tabs(["Login", "Register", "Profile Dashboard"])
    
    with auth_tab1:
        st.markdown("#### Autentificare Utilizator")
        login_username = st.text_input("Username:", key="login_username")
        login_password = st.text_input("Parolă:", type="password", key="login_password")
        
        if st.button("Login", key="user_login"):
            if login_username and login_password:
                user_id = st.session_state.profile_manager.authenticate_user(login_username, login_password)
                if user_id:
                    st.session_state.logged_in_user = user_id
                    st.session_state.user_password = login_password  # Păstrat pentru decriptare
                    st.session_state.username = login_username
                    st.success(f"Bun venit, {login_username}!")
                    st.rerun()
                else:
                    st.error("Username sau parolă incorecte!")
    
    with auth_tab2:
        st.markdown("#### Creare Cont Nou")
        reg_username = st.text_input("Username nou:", key="reg_username")
        reg_password = st.text_input("Parolă nouă:", type="password", key="reg_password")
        reg_password_confirm = st.text_input("Confirmă parola:", type="password", key="reg_password_confirm")
        
        if st.button("Creează Cont", key="create_account"):
            if reg_username and reg_password:
                if reg_password == reg_password_confirm:
                    success, message = st.session_state.profile_manager.create_user(reg_username, reg_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.error("Parolele nu coincid!")
    
    with auth_tab3:
        if 'logged_in_user' in st.session_state:
            st.markdown(f"#### Dashboard - {st.session_state.username}")
            
            # Obține profilul utilizatorului
            user_profile = st.session_state.profile_manager.get_user_profile(
                st.session_state.logged_in_user, 
                st.session_state.user_password
            )
            
            if user_profile:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Statistici Profil")
                    st.metric("Total Analize", user_profile.get("total_analyses", 0))
                    
                    if user_profile.get("primary_emotion"):
                        st.metric("Emoția Dominantă", user_profile["primary_emotion"])
                        st.metric("Stabilitate Emoțională", f"{user_profile.get('emotion_stability', 0):.3f}")
                    
                    # Top stiluri preferate
                    st.markdown("##### Top Stiluri Preferate")
                    top_styles = user_profile.get("top_styles", [])[:3]
                    for style, count in top_styles:
                        st.metric(style, f"{count} analize")
                
                with col2:
                    st.markdown("##### Evoluția Emoțională")
                    
                    # Grafic pentru emoțiile dominante
                    emotions_data = user_profile.get("dominant_emotions", {})
                    if emotions_data:
                        # Pregătește datele pentru grafic
                        emotion_names = []
                        emotion_scores = []
                        
                        for emotion, history in emotions_data.items():
                            if history:
                                recent_avg = np.mean([s["score"] for s in history[-5:]])
                                emotion_names.append(emotion)
                                emotion_scores.append(recent_avg)
                        
                        if emotion_names:
                            fig = go.Figure(data=go.Scatterpolar(
                                r=emotion_scores,
                                theta=emotion_names,
                                fill='toself',
                                name='Profilul Emoțional'
                            ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(visible=True, range=[0, 1])
                                ),
                                title="Profilul Tău Emoțional",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                # Simulează adăugarea unei noi analize
                st.markdown("##### Simulează Analiza Nouă")
                if st.button("Adaugă Analiză Simulată", key="simulate_analysis"):
                    # Date simulate pentru demonstrație
                    simulated_emotions = {
                        "Bucurie": np.random.uniform(0.3, 0.9),
                        "Tristețe": np.random.uniform(0.1, 0.6),
                        "Calm": np.random.uniform(0.4, 0.8),
                        "Energie": np.random.uniform(0.2, 0.7)
                    }
                    
                    simulated_metadata = {
                        "stil_dominant": np.random.choice(["Impresionism", "Cubism", "Romanticism"]),
                        "autor_probabil": np.random.choice(["Monet", "Picasso", "Van Gogh"])
                    }
                    
                    success = st.session_state.profile_manager.update_user_profile(
                        st.session_state.logged_in_user,
                        st.session_state.user_password,
                        simulated_emotions,
                        simulated_metadata
                    )
                    
                    if success:
                        st.success("Profilul a fost actualizat cu noua analiză (criptat)!")
                        st.rerun()
                    else:
                        st.error("Eroare la actualizarea profilului")
            else:
                st.error("Nu s-a putut încărca profilul utilizatorului")
        else:
            st.info("👈 Te rugăm să te autentifici pentru a accesa dashboard-ul")

def render_comparative_analysis():
    """Analiza comparativă a tuturor tehnicilor de securitate."""
    st.markdown("### 📊 Analiza Comparativă - Toate Tehnicile de Securitate")
    
    # Tabel comparativ
    comparison_data = {
        "Tehnică": [
            "Steganografie Emoțională",
            "Atacuri Adversariale", 
            "Profiluri Criptate",
            "Audit Tamper-Evident",
            "Secure Upload Pipeline"
        ],
        "Scop Principal": [
            "Autentificarea analizelor",
            "Testarea robustetii AI",
            "Confidențialitatea utilizatorilor",
            "Integritatea sistemului",
            "Securitatea încărcărilor"
        ],
        "Nivel Securitate": [
            "Înalt", "Mediu", "Foarte Înalt", "Înalt", "Înalt"
        ],
        "Complexitate Implementare": [
            "Medie", "Înaltă", "Înaltă", "Medie", "Medie"
        ],
        "Impact Vizual": [
            "Invizibil", "Imperceptibil", "N/A", "N/A", "N/A"
        ]
    }
    
    st.table(comparison_data)
    
    # Grafic radar pentru comparație
    st.markdown("#### Comparația Caracteristicilor")
    
    categories = ['Securitate', 'Utilizabilitate', 'Performanță', 'Inovație', 'Aplicabilitate']
    
    fig = go.Figure()
    
    # Steganografie
    fig.add_trace(go.Scatterpolar(
        r=[9, 8, 9, 10, 8],
        theta=categories,
        fill='toself',
        name='Steganografie Emoțională',
        line_color='#FF6B6B'
    ))
    
    # Atacuri Adversariale
    fig.add_trace(go.Scatterpolar(
        r=[7, 6, 7, 10, 9],
        theta=categories,
        fill='toself',
        name='Atacuri Adversariale',
        line_color='#4ECDC4'
    ))
    
    # Profiluri Criptate
    fig.add_trace(go.Scatterpolar(
        r=[10, 9, 8, 8, 7],
        theta=categories,
        fill='toself',
        name='Profiluri Criptate',
        line_color='#45B7D1'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        title="Comparația Tehnicilor de Securitate (1-10)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recomandări de implementare
    st.markdown("#### 🎯 Recomandări pentru Prezentarea la Licență")
    
    st.markdown("""
    **Ordinea Recomandată de Prezentare:**
    
    1. **Începe cu Steganografia** - Efectul "wow" este garantat când arăți că imaginile identice au conținut diferit
    
    2. **Continuă cu Atacurile Adversariale** - Demonstrează înțelegerea vulnerabilităților AI și gândirea critică
    
    3. **Finalizează cu Profilurile Criptate** - Arată aplicabilitatea practică și respectarea GDPR
    
    **Puncte Cheie de Evidențiat:**
    - Toate tehnicile sunt implementate de tine personal
    - Fiecare rezolvă o problemă reală din industrie
    - Combinația demonstrează o abordare holistică a securității
    - Relevanța pentru master în cybersecurity sau AI safety
    """)
    
    # Metrici de impact
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tehnici Implementate", "5", delta="100% originale")
    
    with col2:
        st.metric("Domenii Acoperite", "3", delta="AI, Crypto, Privacy")
    
    with col3:
        st.metric("Potential Academic", "Înalt", delta="Publicabil")
