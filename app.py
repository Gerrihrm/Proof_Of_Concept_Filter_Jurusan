import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================================================================
# 1. KONFIGURASI HALAMAN & STREAMLIT LAYOUT
# ==============================================================================
st.set_page_config(
    page_title="Sistem Rekomendasi Karir & Employability ML",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik tampilan UI
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD MODEL PKL & SCALER
# ==============================================================================
@st.cache_resource
def load_ml_assets():
    try:
        models_dict = joblib.load('models_local_fs.pkl')
        scaler_obj = joblib.load('scaler.pkl')
        return models_dict, scaler_obj
    except Exception as e:
        st.error(f"Gagal memuat file .pkl! Pastikan 'models_local_fs.pkl' dan 'scaler.pkl' ada di repositori. Error: {e}")
        return None, None

models_local_fs, scaler = load_ml_assets()

# Ordinal Mappings & Fitur Numerik
ordinal_mappings = {
    'University_Year': {'Freshman': 1, 'Sophomore': 2, 'Junior': 3, 'Senior': 4},
    'Academic_Performance': {'Poor': 1, 'Below Average': 2, 'Average': 3, 'Good': 4, 'Excellent': 5},
    'English_Proficiency': {'Basic': 1, 'Intermediate': 2, 'Advanced': 3}
}

numeric_cols = [
    'Age', 'CGPA', 'Programming_Skill', 'Communication_Skills', 'Teamwork',
    'Problem_Solving', 'Projects_Completed', 'Certifications', 'Internships', 'Hackathons'
]

# ==============================================================================
# 3. INISIALISASI SESSION STATE UNTUK MULTI-STEP WIZARD
# ==============================================================================
if 'step' not in st.session_state:
    st.session_state.step = 1

if 'form_data' not in st.session_state:
    st.session_state.form_data = {
        'Name': '',
        'Age': 21,
        'Gender': 'Male',
        'Major': 'Data Science',
        'University_Year': 'Junior',
        'CGPA': 3.50,
        'Academic_Performance': 'Good',
        'Programming_Skill': 7,
        'Communication_Skills': 7,
        'Problem_Solving': 7,
        'Teamwork': 7,
        'English_Proficiency': 'Intermediate',
        'Projects_Completed': 3,
        'Certifications': 1,
        'Internships': 1,
        'Hackathons': 0,
        'GitHub_Profile': 'Yes',
        'LinkedIn_Profile': 'Yes',
        'Leadership_Experience': 'Yes',
        'Target_Career': ''
    }

# Header Utama
st.markdown("<div class='main-title'>🎓 Sistem Penilaian Employability & Rekomendasi Karir</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Asesmen Kesiapan Kerja Mahasiswa Berbasis Multi-Model Machine Learning</div>", unsafe_allow_html=True)

# Progress Bar
if st.session_state.step <= 4:
    progress_val = st.session_state.step / 4.0
    st.progress(progress_val)
    st.caption(f"Langkah {st.session_state.step} dari 4")

st.divider()

# ==============================================================================
# HALAMAN 1: IDENTITAS DIRI
# ==============================================================================
if st.session_state.step == 1:
    st.subheader("📋 Halaman 1: Identitas Diri")
    st.write("Silakan isi informasi dasar Anda untuk memulai proses asesmen.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.form_data['Name'] = st.text_input(
            "Nama Lengkap", 
            value=st.session_state.form_data['Name'],
            placeholder="Contoh: Budi Santoso"
        )
        st.session_state.form_data['Age'] = st.number_input(
            "Umur (Tahun)", 
            min_value=15, max_value=60, 
            value=int(st.session_state.form_data['Age'])
        )
    
    with col2:
        st.session_state.form_data['Gender'] = st.selectbox(
            "Jenis Kelamin", 
            ['Male', 'Female'], 
            index=0 if st.session_state.form_data['Gender'] == 'Male' else 1
        )
    
    st.write("")
    if st.button("Lanjut ke Informasi Pendidikan ➡️", type="primary"):
        if not st.session_state.form_data['Name'].strip():
            st.warning("Mohon isi nama lengkap terlebih dahulu!")
        else:
            st.session_state.step = 2
            st.rerun()

# ==============================================================================
# HALAMAN 2: INFORMASI PENDIDIKAN
# ==============================================================================
elif st.session_state.step == 2:
    st.subheader("🎓 Halaman 2: Latar Belakang Pendidikan")
    
    col1, col2 = st.columns(2)
    with col1:
        major_options = ['Data Science', 'Computer Science', 'Information Technology', 'Software Engineering', 'Cybersecurity', 'Electrical Engineering', 'Other']
        default_major_idx = major_options.index(st.session_state.form_data['Major']) if st.session_state.form_data['Major'] in major_options else 0
        st.session_state.form_data['Major'] = st.selectbox("Jurusan / Program Studi", major_options, index=default_major_idx)
        
        year_options = ['Freshman', 'Sophomore', 'Junior', 'Senior']
        default_year_idx = year_options.index(st.session_state.form_data['University_Year'])
        st.session_state.form_data['University_Year'] = st.selectbox("Tingkat / Tahun Perkuliahan", year_options, index=default_year_idx)

    with col2:
        st.session_state.form_data['CGPA'] = st.number_input(
            "IPK / CGPA (Skala 4.0)", 
            min_value=0.0, max_value=4.0, 
            value=float(st.session_state.form_data['CGPA']), 
            step=0.01
        )
        
        perf_options = ['Poor', 'Below Average', 'Average', 'Good', 'Excellent']
        default_perf_idx = perf_options.index(st.session_state.form_data['Academic_Performance'])
        st.session_state.form_data['Academic_Performance'] = st.selectbox("Performansi Akademik", perf_options, index=default_perf_idx)

    st.write("")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("⬅️ Kembali"):
            st.session_state.step = 1
            st.rerun()
    with col_next:
        if st.button("Lanjut ke Penilaian Skill ➡️", type="primary"):
            st.session_state.step = 3
            st.rerun()

# ==============================================================================
# HALAMAN 3: PENILAIAN SKILL (SELF-ASSESSMENT)
# ==============================================================================
elif st.session_state.step == 3:
    st.subheader("💡 Halaman 3: Self-Assessment Kompetensi & Skills")
    st.write("Berikan penilaian mandiri terhadap kemampuan teknis dan soft-skill Anda (Skala 1 - 10).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.form_data['Programming_Skill'] = st.slider("Programming Skill", 1, 10, int(st.session_state.form_data['Programming_Skill']))
        st.session_state.form_data['Communication_Skills'] = st.slider("Communication Skill", 1, 10, int(st.session_state.form_data['Communication_Skills']))
        st.session_state.form_data['Problem_Solving'] = st.slider("Problem Solving Skill", 1, 10, int(st.session_state.form_data['Problem_Solving']))

    with col2:
        st.session_state.form_data['Teamwork'] = st.slider("Teamwork & Kolaborasi", 1, 10, int(st.session_state.form_data['Teamwork']))
        eng_options = ['Basic', 'Intermediate', 'Advanced']
        default_eng_idx = eng_options.index(st.session_state.form_data['English_Proficiency'])
        st.session_state.form_data['English_Proficiency'] = st.selectbox("Kemampuan Bahasa Inggris", eng_options, index=default_eng_idx)

    st.write("")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("⬅️ Kembali"):
            st.session_state.step = 2
            st.rerun()
    with col_next:
        if st.button("Lanjut ke Pengalaman ➡️", type="primary"):
            st.session_state.step = 4
            st.rerun()

# ==============================================================================
# HALAMAN 4: RIWAYAT PENGALAMAN & TARGET KARIR
# ==============================================================================
elif st.session_state.step == 4:
    st.subheader("🚀 Halaman 4: Riwayat Portofolio & Posisi Impian")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.form_data['Projects_Completed'] = st.number_input("Jumlah Proyek Selesai", 0, 50, int(st.session_state.form_data['Projects_Completed']))
        st.session_state.form_data['Certifications'] = st.number_input("Jumlah Sertifikasi", 0, 20, int(st.session_state.form_data['Certifications']))
        st.session_state.form_data['Internships'] = st.number_input("Jumlah Magang Ditikuti", 0, 10, int(st.session_state.form_data['Internships']))
        st.session_state.form_data['Hackathons'] = st.number_input("Jumlah Hackathon Ditikuti", 0, 10, int(st.session_state.form_data['Hackathons']))

    with col2:
        st.session_state.form_data['GitHub_Profile'] = st.selectbox("Memiliki Profil GitHub Aktif?", ['Yes', 'No'], index=0 if st.session_state.form_data['GitHub_Profile'] == 'Yes' else 1)
        st.session_state.form_data['LinkedIn_Profile'] = st.selectbox("Memiliki Profil LinkedIn Aktif?", ['Yes', 'No'], index=0 if st.session_state.form_data['LinkedIn_Profile'] == 'Yes' else 1)
        st.session_state.form_data['Leadership_Experience'] = st.selectbox("Pengalaman Organisasi / Leadership?", ['Yes', 'No'], index=0 if st.session_state.form_data['Leadership_Experience'] == 'Yes' else 1)
        
        # Pilihan Target Posisi
        career_options = list(models_local_fs.keys()) if models_local_fs else ['Data Analyst', 'Data Scientist', 'Software Engineer', 'DevOps Engineer']
        st.session_state.form_data['Target_Career'] = st.selectbox("🎯 Pilih Target Posisi Impian Utama Anda", career_options)

    st.write("")
    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("⬅️ Kembali"):
            st.session_state.step = 3
            st.rerun()
    with col_next:
        if st.button("🎯 PROSES & LIHAT HASIL EVALUASI", type="primary"):
            st.session_state.step = 5
            st.rerun()

# ==============================================================================
# HALAMAN 5: HASIL EVALUASI & RANKING REKOMENDASI KARIR
# ==============================================================================
elif st.session_state.step == 5:
    st.subheader("📊 Hasil Evaluasi Employability & Rekomendasi Karir")
    st.write(f"Nama Pendaftar: **{st.session_state.form_data['Name']}** | Jurusan: **{st.session_state.form_data['Major']}**")
    
    # 1. PREPROCESSING INPUT UNTUK INFERENCE
    raw_data = st.session_state.form_data.copy()
    target_career = raw_data.pop('Target_Career')
    student_name = raw_data.pop('Name')
    
    df_input = pd.DataFrame([raw_data])
    
    # A. Ordinal Encoding
    for col, mapping in ordinal_mappings.items():
        if col in df_input.columns:
            df_input[col] = df_input[col].map(mapping)
            
    # B. One-Hot Encoding
    nominal_cols = ['Gender', 'Major', 'GitHub_Profile', 'LinkedIn_Profile', 'Leadership_Experience']
    nominal_cols_present = [c for c in nominal_cols if c in df_input.columns]
    if nominal_cols_present:
        df_input_encoded = pd.get_dummies(df_input, columns=nominal_cols_present)
    else:
        df_input_encoded = df_input.copy()
        
    # C. Scaling Fitur Numerik dengan Penyelarasan Kolom (.reindex)
    if scaler is not None:
        if hasattr(scaler, 'feature_names_in_'):
            scaler_cols = list(scaler.feature_names_in_)
        else:
            scaler_cols = numeric_cols
            
        df_num = df_input_encoded.reindex(columns=scaler_cols, fill_value=0)
        scaled_values = scaler.transform(df_num)
        for i, col in enumerate(scaler_cols):
            df_input_encoded[col] = scaled_values[:, i]
        
    # 2. EVALUASI KE SELURUH SUB-MODEL KARIR
    results = []
    for career, comp in models_local_fs.items():
        rf_model = comp['model']
        threshold = comp['optimal_threshold']
        
        # Ambil daftar & urutan resmi fitur yang dibutuhkan model
        if hasattr(rf_model, 'feature_names_in_'):
            cols_needed = list(rf_model.feature_names_in_)
        elif 'features' in comp:
            cols_needed = list(comp['features'])
        else:
            cols_needed = list(df_input_encoded.columns)
            
        # SOLUSI UTAMA: .reindex() menjamin urutan & nama kolom 100% pas
        df_selected = df_input_encoded.reindex(columns=cols_needed, fill_value=0)
        
        # Prediksi Probabilitas
        prob_placed = float(rf_model.predict_proba(df_selected))
        is_eligible = prob_placed >= threshold
        
        results.append({
            'Bidang Karir': career,
            'Probabilitas Placed': f"{prob_placed * 100:.2f}%",
            'Threshold (F0.5)': f"{threshold * 100:.2f}%",
            'Status Kelolosan': 'ELIGIBLE (Lolos)' if is_eligible else 'NOT ELIGIBLE',
            'Prob_Raw': prob_placed
        })
        
    df_all = pd.DataFrame(results).sort_values(by='Prob_Raw', ascending=False).reset_index(drop=True)
    
    # 3. ISOLASI HASIL POSISI UTAMA
    target_info = df_all[df_all['Bidang Karir'] == target_career]
    
    st.markdown("### 1. Evaluasi Posisi Pilihan Utama")
    if not target_info.empty:
        row = target_info.iloc
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Posisi Target", target_career)
        col_m2.metric("Probabilitas Keberhasilan", row['Probabilitas Placed'])
        col_m3.metric("Threshold Minimum (F0.5)", row['Threshold (F0.5)'])
        
        if "ELIGIBLE" in row['Status Kelolosan']:
            st.success(f"🎉 **KEPUTUSAN: {row['Status Kelolosan']}** — Profil Anda memenuhi standar kualifikasi industri untuk posisi **{target_career}**.")
        else:
            st.error(f"⚠️ **KEPUTUSAN: {row['Status Kelolosan']}** — Profil Anda saat ini belum memenuhi batas ambang minimum untuk posisi **{target_career}**.")
    
    st.divider()
    
    # 4. RANKING REKOMENDASI LINTAS SELURUH BIDANG KARIR
    st.markdown("### 2. Peringkat Rekomendasi Lintas Seluruh Bidang Karir")
    st.write("Berikut adalah hasil perbandingan potensi kesiapan kerja Anda di seluruh rumpun bidang karir:")
    
    st.dataframe(
        df_all.drop(columns=['Prob_Raw']),
        use_container_width=True,
        hide_index=True
    )
    
    st.write("")
    if st.button("🔄 Ulangi Asesmen / Tes Profil Baru", type="primary"):
        st.session_state.step = 1
        st.rerun()

