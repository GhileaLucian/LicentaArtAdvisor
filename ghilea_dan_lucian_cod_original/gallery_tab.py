"""
Componenta pentru tab-ul de galerie în aplicația ArtAdvisor - Versiune Corectată
"""

import streamlit as st
import base64
import os
import datetime
from utils.data_management import get_all_artworks
from config import ALL_EMOTIONS
# Adăugate pentru Upload Securizat și Audit
from utils.uploads_security import secure_store_image
from utils.audit import verify_audit_chain

@st.cache_data(ttl=600, show_spinner=False)  # Cache mai mare - 10 minute
def get_artworks_cached():
    """Versiunea cached pentru obținerea operelor."""
    return get_all_artworks(sort_by_date=True)

@st.cache_data(ttl=180, show_spinner=False)  # Cache pentru 3 minute doar  
def analyze_emotions_distribution(artworks):
    """Analizează distribuția emoțiilor pentru filtre inteligente."""
    emotion_data = {}
    dominant_emotions = []
    
    for artwork in artworks:
        emotions = artwork['metadata'].get('emotii_detectate', {})
        if emotions:
            # Găsește emoția dominantă (cea cu scorul cel mai mare)
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            dominant_emotions.append({
                'artwork': artwork,
                'emotion': dominant_emotion[0],
                'score': dominant_emotion[1]
            })
            
            # Colectează toate emoțiile pentru statistici
            for emotion, score in emotions.items():
                if emotion not in emotion_data:
                    emotion_data[emotion] = []
                emotion_data[emotion].append(score)
    
    return emotion_data, dominant_emotions

@st.cache_data(ttl=300, show_spinner=False)
def get_filter_options(artworks):
    """Obține opțiunile pentru filtre cu cache."""
    all_styles = ['Toate'] + sorted(list(set([
        artwork['metadata'].get('stil_dominant', 'Necunoscut') or 'Necunoscut' 
        for artwork in artworks
    ])))
    
    all_authors = ['Toate'] + sorted(list(set([
        artwork['metadata'].get('autor_probabil', 'Necunoscut') or 'Necunoscut' 
        for artwork in artworks
    ])))
    
    return all_styles, all_authors

def render_gallery_tab():
    """Renderează tab-ul galeriei personale - versiune ULTRA RAPIDĂ."""
    st.title("Galeria Ta")
    st.write("Explorează colecția ta de opere de artă analizate")
    
    # CSS ULTRA-SPECIFIC pentru dimensiuni PERFECT EGALE ale imaginilor
    st.markdown(
        """
        <style>
        /* Toate variantele posibile de selectori pentru imagini în Streamlit */
        div[data-testid="stImage"] > img,
        .stImage > img,
        .stImage img,
        [data-testid="stImage"] img,
        div[data-testid="stImage"] img,
        img[data-testid*="stImage"],
        .element-container img,
        .stColumn img {
            height: 180px !important;
            width: 100% !important;
            max-height: 180px !important;
            min-height: 180px !important;
            max-width: 100% !important;
            min-width: 100% !important;
            object-fit: cover !important;
            object-position: center !important;
            border-radius: 8px !important;
            border: 1px solid #ddd !important;
            box-sizing: border-box !important;
            display: block !important;
        }
        
        /* Container pentru imagini să aibă dimensiune fixă */
        div[data-testid="stImage"],
        .stImage {
            height: 180px !important;
            width: 100% !important;
            overflow: hidden !important;
            border-radius: 8px !important;
            display: block !important;
        }
        
        /* Asigură că toate columnele au comportament consistent */
        .stColumn > div {
            display: flex !important;
            flex-direction: column !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # CSS suplimentar pentru a asigura dimensiuni perfect egale - fallback
    st.markdown(
        """
        <style>
        /* Ultimul nivel de forțare pentru dimensiuni egale */
        * img {
            height: 180px !important;
            width: 100% !important;
            object-fit: cover !important;
            object-position: center !important;
        }
        
        /* Specifig pentru streamlit image containers */
        .stImage, [data-testid="stImage"] {
            height: 180px !important;
            width: 100% !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Folosește versiunea cached
    all_artworks = get_artworks_cached()
    
    if not all_artworks:
        st.info("Galeria ta este goală. Începe să analizezi opere de artă pentru a-ți construi colecția!")
        return

    # Analizează distribuția emoțiilor pentru filtre inteligente
    emotion_data, dominant_emotions = analyze_emotions_distribution(all_artworks)
    all_styles, all_authors = get_filter_options(all_artworks)

    st.subheader("Filtrare Avansată")
    
    # Inițializare filtre îmbunătățite
    if 'gallery_filters' not in st.session_state:
        st.session_state.gallery_filters = {
            'stil_artistic': 'Toate',
            'perioada_timp': 'Toate',
            'emotie_dominanta': 'Toate',  # Nou: emoția principală
            'prag_intensitate': 0.5,      # Nou: prag personalizabil
            'autor_preferat': 'Toate',
        }

    # Layout îmbunătățit pentru filtre - 4 coloane pentru mai multe opțiuni
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write("**Stil & Autor**")
        stil_selected = st.selectbox(
            "Stil Artistic:",
            all_styles,
            index=all_styles.index(st.session_state.gallery_filters['stil_artistic']) if st.session_state.gallery_filters['stil_artistic'] in all_styles else 0,
            key="filter_stil"
        )
        st.session_state.gallery_filters['stil_artistic'] = stil_selected
        
        autor_selected = st.selectbox(
            "Autor:",
            all_authors,
            index=all_authors.index(st.session_state.gallery_filters['autor_preferat']) if st.session_state.gallery_filters['autor_preferat'] in all_authors else 0,
            key="filter_autor"
        )
        st.session_state.gallery_filters['autor_preferat'] = autor_selected
    
    with col2:
        st.write("**Emoție Dominantă**")
        
        # Emoțiile dominante cu numărul de opere
        emotion_counts = {}
        for item in dominant_emotions:
            emotion = item['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        emotion_options = ['Toate'] + [f"{emotion} ({count} opere)" for emotion, count in 
                          sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)]
        
        emotie_dominanta = st.selectbox(
            "Emoția principală:",
            emotion_options,
            key="filter_emotie_dominanta",
            help="Operele unde această emoție are scorul cel mai mare"
        )
        
        # Slider pentru pragul de intensitate
        prag_intensitate = st.slider(
            "Prag intensitate minimă:",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.gallery_filters['prag_intensitate'],
            step=0.05,
            key="filter_prag",
            help="Doar operele cu emoția dominantă peste acest prag"
        )
        st.session_state.gallery_filters['prag_intensitate'] = prag_intensitate
    
    with col3:
        st.write("**Perioada & Calitate**")
        perioada_options = ['Toate', 'Astăzi', 'Săptămâna aceasta', 'Luna aceasta', 'Ultimele 3 luni', 'Mai vechi']
        perioada_selected = st.selectbox(
            "Perioada analizei:",
            perioada_options,
            index=perioada_options.index(st.session_state.gallery_filters['perioada_timp']) if st.session_state.gallery_filters['perioada_timp'] in perioada_options else 0,
            key="filter_perioada"
        )
        st.session_state.gallery_filters['perioada_timp'] = perioada_selected
        
        # Filtru nou pentru calitatea analizei
        calitate_options = ['Toate', 'Încredere mare (>80%)', 'Încredere medie (50-80%)', 'Încredere mică (<50%)']
        calitate_selected = st.selectbox(
            "Calitatea identificării:",
            calitate_options,
            key="filter_calitate",
            help="Bazat pe încrederea modelului în identificarea stilului/autorului"
        )
    
    with col4:
        st.write("**Acțiuni**")
        
        # Butoane de acțiune
        if st.button("Resetează Filtrele", use_container_width=True):
            st.session_state.gallery_filters = {
                'stil_artistic': 'Toate',
                'perioada_timp': 'Toate',
                'emotie_dominanta': 'Toate',
                'prag_intensitate': 0.5,
                'autor_preferat': 'Toate',
            }
            st.rerun()
        
        if st.button("Statistici Rapide", use_container_width=True):
            st.session_state.show_stats = not st.session_state.get('show_stats', False)
        
        # Selector pentru numărul de opere pe rând (optimizare performanță)
        opere_per_rand = st.selectbox(
            "Opere pe rând:",
            [3, 4, 5, 6],
            index=2,  # default 3 pentru viteză maximă
            key="opere_per_rand",
            help="Mai puține = încărcare MULT mai rapidă"
        )

    # Aplicarea filtrelor îmbunătățite
    now = datetime.datetime.now()
    artworks_to_show = all_artworks.copy()
    
    # Filtrare după stil artistic
    if stil_selected != 'Toate':
        artworks_to_show = [artwork for artwork in artworks_to_show 
            if (artwork['metadata'].get('stil_dominant', 'Necunoscut') or 'Necunoscut') == stil_selected]
    
    # Filtrare după autor
    if autor_selected != 'Toate':
        artworks_to_show = [artwork for artwork in artworks_to_show 
            if (artwork['metadata'].get('autor_probabil', 'Necunoscut') or 'Necunoscut') == autor_selected]
    
    # Filtrare după emoția dominantă (îmbunătățită)
    if emotie_dominanta != 'Toate':
        emotion_name = emotie_dominanta.split(' (')[0]  # Extrage numele emoției
        filtered_by_emotion = []
        for artwork in artworks_to_show:
            emotions = artwork['metadata'].get('emotii_detectate', {})
            if emotions:
                # Găsește emoția cu scorul cel mai mare
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                if (dominant_emotion[0] == emotion_name and 
                    dominant_emotion[1] >= prag_intensitate):
                    filtered_by_emotion.append(artwork)
        artworks_to_show = filtered_by_emotion
    
    # Filtrare după perioada de timp (îmbunătățită)
    if perioada_selected != 'Toate':
        filtered_by_time = []
        for artwork in artworks_to_show:
            data_analiza = artwork['metadata'].get('data_analiza', '')
            try:
                if data_analiza:
                    data_str = data_analiza.split(' ')[0].replace('/', '.').replace('-', '.')
                    data_obj = datetime.datetime.strptime(data_str, '%d.%m.%Y')
                    diff_days = (now - data_obj).days
                    
                    if perioada_selected == 'Astăzi' and diff_days == 0:
                        filtered_by_time.append(artwork)
                    elif perioada_selected == 'Săptămâna aceasta' and diff_days <= 7:
                        filtered_by_time.append(artwork)
                    elif perioada_selected == 'Luna aceasta' and diff_days <= 30:
                        filtered_by_time.append(artwork)
                    elif perioada_selected == 'Ultimele 3 luni' and diff_days <= 90:
                        filtered_by_time.append(artwork)
                    elif perioada_selected == 'Mai vechi' and diff_days > 90:
                        filtered_by_time.append(artwork)
                else:
                    if perioada_selected == 'Mai vechi':
                        filtered_by_time.append(artwork)
            except Exception:
                if perioada_selected == 'Mai vechi':
                    filtered_by_time.append(artwork)
        artworks_to_show = filtered_by_time
    
    # Filtrare după calitatea identificării (nou)
    if calitate_selected != 'Toate':
        filtered_by_quality = []
        for artwork in artworks_to_show:
            stil_scor = artwork['metadata'].get('stil_scor', 0)
            autor_scor = artwork['metadata'].get('autor_scor', 0)
            max_confidence = max(stil_scor, autor_scor)
            
            if calitate_selected == 'Încredere mare (>80%)' and max_confidence > 0.8:
                filtered_by_quality.append(artwork)
            elif calitate_selected == 'Încredere medie (50-80%)' and 0.5 <= max_confidence <= 0.8:
                filtered_by_quality.append(artwork)
            elif calitate_selected == 'Încredere mică (<50%)' and max_confidence < 0.5:
                filtered_by_quality.append(artwork)
        artworks_to_show = filtered_by_quality

    # Afișare statistici rapide dacă sunt solicitate
    if st.session_state.get('show_stats', False):
        with st.expander("Statistici Galerie", expanded=True):
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("Total opere", len(all_artworks))
                st.metric("Opere filtrate", len(artworks_to_show))
            
            with stat_col2:
                if emotion_data:
                    most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])
                    st.metric("Emoția cea mai frecventă", most_common_emotion[0])
                    st.metric("Număr opere cu această emoție", most_common_emotion[1])
            
            with stat_col3:
                unique_styles = len(set([artwork['metadata'].get('stil_dominant', 'Necunoscut') 
                                       for artwork in all_artworks]))
                st.metric("Stiluri unice", unique_styles)
                unique_authors = len(set([artwork['metadata'].get('autor_probabil', 'Necunoscut') 
                                        for artwork in all_artworks]))
                st.metric("Autori unici", unique_authors)
            
            with stat_col4:
                if artworks_to_show:
                    avg_confidence = sum([max(artwork['metadata'].get('stil_scor', 0), 
                                            artwork['metadata'].get('autor_scor', 0)) 
                                        for artwork in artworks_to_show]) / len(artworks_to_show)
                    st.metric("Încredere medie", f"{avg_confidence:.1%}")

    # Header cu informații despre filtrare
    active_filters_count = 0
    filter_descriptions = []
    
    if stil_selected != 'Toate':
        active_filters_count += 1
        filter_descriptions.append(f"Stil: {stil_selected}")
    if autor_selected != 'Toate':
        active_filters_count += 1
        filter_descriptions.append(f"Autor: {autor_selected}")
    if emotie_dominanta != 'Toate':
        active_filters_count += 1
        filter_descriptions.append(f"Emoție: {emotie_dominanta}")
    if perioada_selected != 'Toate':
        active_filters_count += 1
        filter_descriptions.append(f"Perioada: {perioada_selected}")
    if calitate_selected != 'Toate':
        active_filters_count += 1
        filter_descriptions.append(f"Calitate: {calitate_selected}")
    if prag_intensitate > 0.5:
        active_filters_count += 1
        filter_descriptions.append(f"Intensitate: >{prag_intensitate:.0%}")
    
    filter_text = f"({active_filters_count} filtre active)" if active_filters_count > 0 else "(niciun filtru activ)"

    # Afișare rezultate filtrare - simplu și rapid
    st.write(f"**{len(artworks_to_show)} opere găsite din {len(all_artworks)}** {filter_text}")
    
    if artworks_to_show:
        # Stocăm informațiile pentru dialog în session_state
        if 'artwork_details' not in st.session_state:
            st.session_state.artwork_details = {}
        
        # Afișare simplă și rapidă
        images_per_row = opere_per_rand
        
        # Împărțim operele în rânduri
        for i in range(0, len(artworks_to_show), images_per_row):
            cols = st.columns(images_per_row)
            
            # Pentru fiecare coloană din rândul curent
            for j, col in enumerate(cols):
                if i + j < len(artworks_to_show):
                    artwork = artworks_to_show[i + j]
                    metadata = artwork['metadata']
                    image_path = artwork['image_path']
                    image_id = metadata.get('timestamp', f"img_{i+j}")
                    
                    # Stocăm detaliile
                    st.session_state.artwork_details[image_id] = {
                        'metadata': metadata,
                        'image_path': image_path
                    }
                    
                    # Obținem informațiile pentru afișare
                    stil = metadata.get('stil_dominant', 'Stil necunoscut')
                    autor = metadata.get('autor_probabil', 'Autor necunoscut')
                    
                    with col:
                        # SOLUȚIE ULTRA RAPIDĂ - imagini perfect egale
                        try:
                            # Imaginile vor fi perfect egale prin CSS forțat
                            st.image(
                                image_path, 
                                use_container_width=True, 
                                caption=stil
                            )
                        except Exception as e:
                            st.error(f"Eroare la încărcarea imaginii: {os.path.basename(image_path)}")
                        
                        # Buton pentru detalii - compact
                        if st.button("📋 Detalii", key=f"btn_{image_id}", use_container_width=True):
                            st.session_state[f"show_details_{image_id}"] = True
        
        # Afișăm detaliile pentru imaginea selectată
        for artwork in artworks_to_show:
            metadata = artwork['metadata']
            image_path = artwork['image_path']
            image_id = metadata.get('timestamp', '')
            
            if st.session_state.get(f"show_details_{image_id}", False):
                with st.expander(f"Detalii: {metadata.get('stil_dominant', 'Operă Necunoscută')}", expanded=True):
                    # Buton pentru a închide detaliile
                    if st.button("Închide", key=f"close_{image_id}"):
                        st.session_state[f"show_details_{image_id}"] = False
                        st.rerun()
                    
                    # Layoutul principal în două coloane
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.image(image_path, use_container_width=True)
                        
                        # Info card pentru detalii principale
                        st.markdown(
                            f'''
                            <div class="info-card">
                                <small>Analizată la:</small>
                                <p style="margin: 0; color: #f7c873;">{metadata.get('data_analiza', 'N/A')}</p>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.header(f"{metadata.get('stil_dominant', 'Stil Necunoscut')}")
                        st.subheader(f"de {metadata.get('autor_probabil', 'Autor Necunoscut')}")
                        
                        # Scorul de încredere pentru stil și autor
                        col_stil, col_autor = st.columns(2)
                        with col_stil:
                            st.metric(
                                "Încredere Stil",
                                f"{metadata.get('stil_scor', 0):.1%}",
                                delta_color="off"
                            )
                        with col_autor:
                            st.metric(
                                "Încredere Autor",
                                f"{metadata.get('autor_scor', 0):.1%}",
                                delta_color="off"
                            )
                        
                        # Descrierea narativă
                        if metadata.get('descriere_narrativa') and metadata['descriere_narrativa'] != 'N/A':
                            st.markdown(
                                f'<div style="padding: 1rem; border-left: 3px solid #f7c873; margin: 1rem 0;">'
                                f'<em>{metadata["descriere_narrativa"]}</em></div>',
                                unsafe_allow_html=True
                            )
                        
                        # Emoții detectate simplu
                        st.subheader("Emoții Predominante")
                        emotions = metadata.get('emotii_detectate', {})
                        if emotions:
                            # Sortează emoțiile după scor
                            sorted_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
                            for emotion, score in sorted_emotions[:5]:
                                st.text(f"{emotion}: {score:.1%}")
                        else:
                            st.info("Nu s-au detectat emoții.")
    
    else:
        # Mesaj pentru galerie goală - simplu
        st.warning("Nicio operă găsită cu aceste filtre. Încearcă să modifici criteriile sau resetează filtrele.")
    
    # Încărcare sigură imagine (Secure Upload Pipeline)
    with st.expander("Încărcare sigură imagine", expanded=False):
        up = st.file_uploader(
            "Încarcă imagine (JPG/PNG/WebP, max 10MB)",
            type=["jpg", "jpeg", "png", "webp"],
            key="secure_upload_file"
        )
        c1, c2 = st.columns(2)
        with c1:
            if up and st.button("Salvează în galerie", key="btn_secure_save", use_container_width=True):
                try:
                    res = secure_store_image(up)
                    msg = f"Imagine salvată: {os.path.basename(res['image_path'])}"
                    if res.get("duplicate_of"):
                        msg += f" (atenție: seamănă cu {res['duplicate_of']})"
                    st.success(msg)
                    st.caption(f"SHA-256: {res.get('sha256','-')}  |  pHash: {res.get('phash','-') or '-'}")
                    # Reîncarcă lista de opere după upload
                    st.rerun()
                except Exception as e:
                    st.error(f"Eroare încărcare: {e}")
        with c2:
            if st.button("Verifică integritatea audit log", key="btn_verify_audit", use_container_width=True):
                ok = verify_audit_chain()
                if ok:
                    st.success("Lanțul de audit este valid.")
                else:
                    st.error("Lanțul de audit este INVALID. Verificați fișierul logs/audit_log.csv.")
