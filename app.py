import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm
import base64
from io import BytesIO
import re

# ===================================================================================
# 1. تعريف مسارات الملفات والألوان الموحدة (Palette)
# ===================================================================================

# ألوان المشروع الموحدة (البنفسجي بدرجاته)
PURPLE_DARK = "#4B0082"  # بنفسجي غامق (للعناوين، محاور الرسم، لون الزر الأساسي)
PURPLE_BUTTON = "#4B0082" # لون الزر الأساسي (تم توحيده مع الغامق)
PURPLE_LIGHT = "#EBE0FF" # بنفسجي فاتح (لتأثير التمرير hover)
GRAY_TEXT = "#333333"    # رمادي غامق للنصوص العادية وقيم المقاييس (st.metric)
WHITE_TEXT = "white"     # لون أبيض للنص داخل الأزرار والمودال

CONTACT_EMAIL = "info@chemical-spectra.com" 

MOLECULES_INFO = {
    "بنزين (Benzene)": {
        "file": "outputBenzene.txt",
        "explanation": "يتميز البنزين بامتصاص ضعيف نسبيًا في منطقة فوق البنفسجية العميقة. يرجع ذلك إلى **التماثل العالي** للجزيء، مما يجعل الانتقالات الإلكترونية من النوع $\\pi \\rightarrow \\pi^*$ غير فعالة (ممنوعة) وفق قواعد الاختيار.",
        "color": "#6A5ACD" 
    },
    "فينول (Phenol)": {
        "file": "outputPhenol.txt",
        "explanation": "مجموعة الهيدروكسيل ($\text{-OH}$) في الفينول هي مجموعة **مانحة للإلكترونات (Auxochrome)**. هذا يقلل من الفجوة الطاقية بين المدارات، ويسهل الانتقال الإلكتروني، مما يسبب **انزياحًا باتجاه الأطوال الموجية الأطول (Bathochromic Shift)** مقارنة بالبنزين.",
        "color": "#9370DB" 
    },
    "نيتروبنزين (Nitrobenzene)": {
        "file": "outputNitrobenzene.txt",
        "explanation": "مجموعة النيترو ($\text{-NO}_2$) هي مجموعة **ساحبة قوية للإلكترونات (Chromophore)**. هذا يخلق نظام **ناقل للشحنة ($\text{CT}$ Band)** قوي للغاية بين حلقة البنزين ومجموعة النيترو، مما ينتج قمة امتصاص كبيرة جدًا ومنزاحة بوضوح باتجاه الأطوال الموجية الأطول.",
        "color": "#8A2BE2" 
    }
}

# ===================================================================================
# 2. CSS المخصص للتصميم الاحترافي
# ===================================================================================

CUSTOM_CSS = f"""
<style>
/* 1. خلفية الصفحة */
.stApp {{
    background-color: {PURPLE_LIGHT};
    color: {GRAY_TEXT};
}}

/* 2. توحيد شكل جميع الأزرار */
div.stButton > button, 
div[data-testid="stDownloadButton"] > button,
button[data-testid="baseButton-secondary"],
div[data-testid="stPopover"] > button,
[data-testid="stPopover"] button {{
    background-color: {PURPLE_DARK} !important;
    color: {WHITE_TEXT} !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    border: none !important;
    transition: background-color 0.3s, color 0.3s, border 0.3s !important;
    font-weight: bold !important;
    height: 3.5em !important;
    min-height: 3.5em !important;
    width: 100% !important;
}}

/* 3. تأثير تمرير الماوس على الأزرار (Hover) */
div.stButton > button:hover,
div[data-testid="stDownloadButton"] > button:hover,
button[data-testid="baseButton-secondary"]:hover,
div[data-testid="stPopover"] > button:hover,
[data-testid="stPopover"] button:hover {{
    background-color: {PURPLE_LIGHT} !important;
    color: {GRAY_TEXT} !important;
    border: 1px solid {PURPLE_DARK} !important;
}}

/* 4. تأكيد إضافي على زر الـ popover */
button[kind="secondary"] {{
    background-color: {PURPLE_DARK} !important;
    color: {WHITE_TEXT} !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    height: 3.5em !important;
    min-height: 3.5em !important;
}}

button[kind="secondary"]:hover {{
    background-color: {PURPLE_LIGHT} !important;
    color: {GRAY_TEXT} !important;
}}

/* 4. تنسيق الـ Popover (المودال) */
/* خلفية المودال */
div[role="dialog"] {{
    background-color: {PURPLE_DARK} !important;
    border-radius: 8px !important;
    padding: 20px !important;
}}

/* النصوص داخل المودال */
div[role="dialog"] h4,
div[role="dialog"] p {{
    color: {WHITE_TEXT} !important;
}}

/* 5. تنسيق القائمة الجانبية (Radio Button) */
[data-testid="stSidebar"] {{
    background-color: {PURPLE_LIGHT};
}}

[data-testid="stSidebar"] label[data-baseweb="radio"] {{
    border-radius: 8px;
    padding: 8px 12px;
    transition: background-color 0.3s;
}}

[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {{
    background-color: {GRAY_TEXT} !important;
    color: white !important;
}}

[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) > div {{
    color: white !important;
}}

[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {{
    background-color: rgba(75, 0, 130, 0.1);
}}

/* 6. تنسيق العناوين والمقاييس */
h1, h2, h3, h4 {{
    color: {PURPLE_DARK} !important;
}}

[data-testid="stMetricLabel"] {{
    color: {PURPLE_DARK} !important;
    font-weight: bold;
}}

[data-testid="stMetricValue"] {{
    color: {GRAY_TEXT} !important;
}}

/* 7. إخفاء الترويسة وال Footer */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
</style>
"""

# ===================================================================================
# 3. وظيفة قراءة البيانات من ملفات ORCA
# ===================================================================================

@st.cache_data
def read_orca_data(file_path):
    wavelengths = []
    f_osc_values = []
    dipole_moment = None
    start_reading_spectrum = False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None, 0.0 

    for line in lines:
        if 'Magnitude (Debye)' in line:
            match = re.search(r':\s*(\d+\.\d+)', line)
            if match:
                try:
                    dipole_moment = float(match.group(1))
                except ValueError:
                    pass
        if 'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' in line:
            start_reading_spectrum = True
            continue
        if start_reading_spectrum:
            if '-----------------------------------------------------------------------------' in line or 'ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS' in line:
                if len(wavelengths) > 0:
                     break
            if re.match(r'^\s*\d+\s+[\d\.]+', line):
                try:
                    parts = line.split()
                    wavelength = float(parts[2])
                    f_osc = float(parts[3])
                    
                    if f_osc > 0.00001: 
                        wavelengths.append(wavelength)
                        f_osc_values.append(f_osc)
                except (ValueError, IndexError):
                    continue
            
    if not wavelengths:
        return None, dipole_moment if dipole_moment is not None else 0.0

    df_peaks = pd.DataFrame({'Wavelength (nm)': wavelengths, 'Osc. Strength (f)': f_osc_values})
    return df_peaks, dipole_moment if dipole_moment is not None else 0.0

# ===================================================================================
# 4. وظيفة محاكاة ورسم الطيف
# ===================================================================================

def simulate_spectrum(df_peaks, fwhm=20, min_w=180, max_w=450, num_points=1000):
    if df_peaks is None or df_peaks.empty:
        return pd.DataFrame({'Wavelength (nm)': np.linspace(min_w, max_w, num_points), 'Normalized Absorption': np.zeros(num_points)})

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    wavelength_range = np.linspace(min_w, max_w, num_points)
    absorption_spectrum = np.zeros_like(wavelength_range, dtype=float)

    for index, row in df_peaks.iterrows():
        center = row['Wavelength (nm)']
        f_osc = row['Osc. Strength (f)']
        gaussian_curve = norm.pdf(wavelength_range, center, sigma)
        absorption_spectrum += f_osc * gaussian_curve

    max_absorption = np.max(absorption_spectrum)
    if max_absorption > 0:
        normalized_spectrum = absorption_spectrum / max_absorption
    else:
        normalized_spectrum = absorption_spectrum
        
    df_spectrum = pd.DataFrame({
        'Wavelength (nm)': wavelength_range,
        'Normalized Absorption': normalized_spectrum
    })
    
    return df_spectrum

# ===================================================================================
# 5. وظيفة إنشاء ملف Excel موحد لجميع المركبات
# ===================================================================================

@st.cache_data(show_spinner=False)
def generate_multi_excel_file(molecules_info):
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for mol_name, info in molecules_info.items():
            df_peaks, dipole_moment = read_orca_data(info["file"])
            
            if df_peaks is not None and not df_peaks.empty:
                df_spectrum = simulate_spectrum(df_peaks)
                
                df_peaks_export = df_peaks.rename(columns={'Wavelength (nm)': 'الطول الموجي (nm)', 'Osc. Strength (f)': 'شدة المذبذب'})
                df_peaks_export['عزم ثنائي القطب (Debye)'] = dipole_moment
                df_peaks_export.to_excel(writer, sheet_name=f"{mol_name} - القمم", index=False)
                
                df_spectrum_export = df_spectrum.rename(columns={'Wavelength (nm)': 'الطول الموجي (nm)', 'Normalized Absorption': 'الامتصاص المُعاير'})
                df_spectrum_export.to_excel(writer, sheet_name=f"{mol_name} - الطيف", index=False)
    
    return output.getvalue()

# ===================================================================================
# 6. وظيفة إنشاء رابط التحميل (Excel للمركب الواحد)
# ===================================================================================

def to_excel_download_link(df_spectrum, df_peaks, molecule_name):
    """تنشئ رابط تحميل لملف Excel للمركب الواحد."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_peaks.to_excel(writer, sheet_name='القمم_الرئيسية', index=False)
        df_spectrum.to_excel(writer, sheet_name='منحنى_الطيف_المحاكى', index=False)
    
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    
    href = f"""
    <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" 
       download="Spectral_Data_{molecule_name}.xlsx" 
       class="download-btn-link">
       📥 تحميل بيانات {molecule_name}
    </a>
    <style>
    .download-btn-link {{
        background-color: {PURPLE_BUTTON}; 
        color: white; 
        padding: 10px 20px; 
        border-radius: 8px; 
        text-decoration: none; 
        display: inline-block;
        font-weight: bold;
        transition: background-color 0.3s, color 0.3s, border 0.3s;
    }}
    .download-btn-link:hover {{
        background-color: {PURPLE_LIGHT};
        color: {GRAY_TEXT};
        border: 1px solid {PURPLE_DARK};
    }}
    </style>
    """
    return href

# ===================================================================================
# 7. التطبيق الرئيسي Streamlit
# ===================================================================================

def main():
    # 7.1. إعدادات الصفحة وتطبيق الـ CSS
    st.set_page_config(
        page_title="محاكاة تحليل الامتصاص الضوئي للجزيئات العضوية",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # 7.2. عنوان المشروع
    st.markdown(f"<h1 style='text-align: center;'>محاكاة تحليل الامتصاص الضوئي للجزيئات العضوية</h1>", unsafe_allow_html=True)
    
    # 7.3. شريط التنقل (Navigation Bar)
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        st.button("الرئيسية 🏠", key="home_btn", use_container_width=True)
        
    with nav_col2:
        with st.popover("عن المشروع ℹ️", use_container_width=True):
            st.markdown("#### هدف المشروع:")
            st.markdown("""
            يهدف هذا المشروع إلى محاكاة وتحليل الأطياف المرئية وفوق البنفسجية (UV-Vis) 
            للمركبات العضوية العطرية باستخدام نتائج حسابات الكيمياء الكمومية (TD-DFT)، 
            وتوضيح العلاقة بين البنية الكيميائية وخصائص الامتصاص الضوئي.
            
            يتم استخلاص البيانات من ملفات إخراج ORCA لمحاكاة الطيف باستخدام منحنيات غاوس.
            """)

    multi_excel_data = generate_multi_excel_file(MOLECULES_INFO)
    with nav_col3:
        st.download_button(
            label="تحميل النتائج 💾",
            data=multi_excel_data,
            file_name="Spectral_Analysis_All_Molecules.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_all_btn",
            use_container_width=True
        )

    with nav_col4:
        mailto_button_html = f"""
        <a href="mailto:{CONTACT_EMAIL}" style="text-decoration: none; width: 100%; display: block;">
            <button style="
                width: 100%; 
                height: 3.5em; 
                background-color: {PURPLE_DARK}; 
                color: {WHITE_TEXT}; 
                border-radius: 8px; 
                font-weight: bold;
                border: none;
                padding: 10px 20px;
                cursor: pointer;
                transition: background-color 0.3s, color 0.3s, border 0.3s;
            " 
            onmouseover="this.style.backgroundColor='{PURPLE_LIGHT}'; this.style.color='{GRAY_TEXT}'; this.style.border='1px solid {PURPLE_DARK}';"
            onmouseout="this.style.backgroundColor='{PURPLE_DARK}'; this.style.color='{WHITE_TEXT}'; this.style.border='none';"
            >
                تواصل معنا 📧
            </button>
        </a>
        """
        st.markdown(mailto_button_html, unsafe_allow_html=True)

    st.markdown("---")

    # 7.4. القائمة الجانبية لاختيار المركب
    st.sidebar.header("اختر المركب للتحليل")
    molecule_name = st.sidebar.radio(
        "", 
        list(MOLECULES_INFO.keys()),
        index=0 
    )

    # 7.5. معالجة بيانات المركب وعرضها
    info = MOLECULES_INFO[molecule_name]
    file_path = info["file"]
    
    df_peaks, dipole_moment = read_orca_data(file_path)
    
    if df_peaks is None or df_peaks.empty:
        st.error(f"❌ خطأ: لم يتم العثور على ملف البيانات `{file_path}` أو لا يحتوي على جدول طيف.")
        st.stop()

    df_spectrum = simulate_spectrum(df_peaks)
    strongest_peak = df_peaks.loc[df_peaks['Osc. Strength (f)'].idxmax()]
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"## 🧪 بيانات تحليل المركب")
        st.markdown("---")
        
        st.metric(
            label="أقوى طول موجي ($\lambda_{max}$)",
            value=f"{strongest_peak['Wavelength (nm)']:.1f} nm"
        )
        st.metric(
            label="قيمة الامتصاص ($f_{osc}$)",
            value=f"{strongest_peak['Osc. Strength (f)']:.3f}"
        )
        st.metric(
            label="عزم ثنائي القطب (Dipole Moment)",
            value=f"{dipole_moment:.2f} Debye"
        )
        
        st.markdown("---")
        st.markdown(to_excel_download_link(df_spectrum, df_peaks, molecule_name.replace(' ', '_')), unsafe_allow_html=True)
        
    with col2:
        st.markdown("## 📊 الطيف الممتص (UV-Vis) - تفاعلي")
        
        fig = px.line(
            df_spectrum, 
            x='Wavelength (nm)', 
            y='Normalized Absorption',
            title=f"طيف الامتصاص المحاكى لـ {molecule_name}",
            labels={'Wavelength (nm)': 'الطول الموجي (nm)', 'Normalized Absorption': 'الامتصاص المُعاير'}
        )
        
        fig.update_xaxes(autorange="reversed") 
        
        for index, row in df_peaks.iterrows():
             fig.add_vline(x=row['Wavelength (nm)'], line_dash="dash", line_color=info['color'], 
                           annotation_text=f"{row['Wavelength (nm)']:.1f} nm", 
                           annotation_position="top left", annotation_font_size=10)

        fig.update_layout(
            yaxis_title='الامتصاص المُعاير',
            xaxis_title='الطول الموجي (nm)',
            title_font_color=PURPLE_DARK, 
            
            xaxis=dict(
                showgrid=True, gridcolor='lightgray',
                linecolor=PURPLE_DARK, 
                tickfont=dict(color=GRAY_TEXT),
                title_font=dict(color=PURPLE_DARK)
            ),
            yaxis=dict(
                showgrid=True, gridcolor='lightgray',
                linecolor=PURPLE_DARK, 
                tickfont=dict(color=GRAY_TEXT),
                title_font=dict(color=PURPLE_DARK)
            ),
            
            plot_bgcolor='white',
            paper_bgcolor='rgba(0,0,0,0)', 
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("---")
    st.markdown("## 📋 جدول بالقيم المستخرجة")
    st.dataframe(df_peaks.rename(columns={'Wavelength (nm)': 'الطول الموجي (nm)', 'Osc. Strength (f)': 'قيمة الامتصاص (a.u)'}), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### ملاحظة توضيحية:")
    st.markdown(f"**العلاقة بين التركيب الكيميائي للمركب وشدة امتصاصه للضوء:** {info['explanation']}")


if __name__ == "__main__":
    main()